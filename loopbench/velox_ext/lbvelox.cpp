// loopbench Velox extension.
//
// * Registers loopbench scalar functions (word_ngram_hashes, minhash_bands) and the
//   Parquet reader/writer into the shared libvelox.so, so PyVelox plans see them too.
// * Executes plans described as JSON step lists with the full exec PlanBuilder
//   (partial/final aggregation over local exchanges, semi/anti joins, local merge),
//   multi-threaded, returning results through the Arrow C data interface.
//
// Plan JSON: a list of steps; the first is a scan, join steps carry a "build" plan.
//   {"op": "scan", "files": [...], "columns": [[name, type], ...], "filter": expr?}
//   {"op": "filter", "expr": e}            {"op": "project", "exprs": [...]}
//   {"op": "agg", "keys": [...], "aggs": [...]}    (parallel partial -> exchange -> final)
//   {"op": "join", "type": inner|left|semi|anti, "left": [...], "right": [...],
//    "build": plan, "output": [...], "filter": e?}
//   {"op": "unnest", "replicate": [...], "columns": [...], "names": [...]}
//   {"op": "order_by", "keys": [...]}      {"op": "limit", "n": k}
//   {"op": "write", "dir": path}           (Parquet, one file per driver)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <filesystem>

#include <execinfo.h>
#include <csignal>
#include <unistd.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/json.h>

#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/Macros.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/arrow/Bridge.h"

#include "TextSig.h"

namespace loopbench {

namespace py = pybind11;
using namespace facebook::velox;

namespace {

// word_ngram_hashes(text, n) -> array(bigint): one hash per n-word window.
template <typename TExec>
struct WordNgramHashesFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  std::vector<uint64_t> words_;
  std::vector<uint64_t> grams_;

  template <typename TInt>
  void call(out_type<Array<int64_t>>& out, const arg_type<Varchar>& text, const TInt& n) {
    wordHashes(std::string_view(text.data(), text.size()), words_);
    ngramHashes(words_, n, grams_);
    out.reserve(grams_.size());
    for (uint64_t h : grams_) {
      out.push_back(static_cast<int64_t>(h));
    }
  }
};

// minhash_bands(text, bands, rows, shingle_words, seed) -> array(bigint): one LSH
// band hash per band (empty if the text has fewer than shingle_words words).
template <typename TExec>
struct MinhashBandsFunction {
  VELOX_DEFINE_FUNCTION_TYPES(TExec);

  std::vector<uint64_t> words_;
  std::vector<uint64_t> shingles_;
  std::vector<uint64_t> bands_;

  template <typename TInt>
  void call(
      out_type<Array<int64_t>>& out, const arg_type<Varchar>& text, const TInt& bands,
      const TInt& rows, const TInt& shingle, const int64_t& seed) {
    wordHashes(std::string_view(text.data(), text.size()), words_);
    ngramHashes(words_, shingle, shingles_);
    minhashBands(shingles_, bands, rows, static_cast<uint64_t>(seed), bands_);
    out.reserve(bands_.size());
    for (uint64_t h : bands_) {
      out.push_back(static_cast<int64_t>(h));
    }
  }
};

void registerFunctions() {
  parquet::registerParquetReaderFactory();
  parquet::registerParquetWriterFactory();
  registerFunction<WordNgramHashesFunction, Array<int64_t>, Varchar, int32_t>({"word_ngram_hashes"});
  registerFunction<WordNgramHashesFunction, Array<int64_t>, Varchar, int64_t>({"word_ngram_hashes"});
  registerFunction<MinhashBandsFunction, Array<int64_t>, Varchar, int32_t, int32_t, int32_t, int64_t>(
      {"minhash_bands"});
  registerFunction<MinhashBandsFunction, Array<int64_t>, Varchar, int64_t, int64_t, int64_t, int64_t>(
      {"minhash_bands"});
}

const std::string kConnectorId{exec::test::PlanBuilder::kHiveDefaultConnectorId};

struct Runtime {
  std::shared_ptr<folly::CPUThreadPoolExecutor> executor;
  // Output vectors are exported to Arrow and may be released by Python long after the
  // query; their buffers must come from a pool that lives for the whole process.
  std::shared_ptr<memory::MemoryPool> outputPool;
};

Runtime& runtime() {
  static Runtime rt;
  return rt;
}

void init(int threads) {
  static bool done = false;
  if (done) {
    return;
  }
  done = true;
  if (!memory::MemoryManager::testInstance()) {
    memory::initializeMemoryManager(memory::MemoryManager::Options{});
    if (getenv("LBVELOX_BACKTRACE")) {
      auto handler = [](int) {
        void* frames[64];
        backtrace_symbols_fd(frames, backtrace(frames, 64), STDERR_FILENO);
        _exit(134);
      };
      signal(SIGABRT, handler);
      signal(SIGSEGV, handler);
    }
  }
  filesystems::registerLocalFileSystem();
  dwio::common::LocalFileSink::registerFactory();
  parse::registerTypeResolver();
  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();
  registerFunctions();
  auto& rt = runtime();
  rt.executor = std::make_shared<folly::CPUThreadPoolExecutor>(threads);
  rt.outputPool = memory::memoryManager()->addLeafPool("loopbench_output");
  if (!connector::ConnectorRegistry::tryGet(kConnectorId)) {
    auto config = std::make_shared<config::ConfigBase>(std::unordered_map<std::string, std::string>{});
    connector::hive::HiveConnectorFactory factory;
    connector::ConnectorRegistry::global().insert(
        kConnectorId, factory.newConnector(kConnectorId, config, rt.executor.get()));
  }
}

TypePtr parseType(const std::string& t) {
  static const std::unordered_map<std::string, TypePtr> kTypes = {
      {"varchar", VARCHAR()}, {"bigint", BIGINT()}, {"integer", INTEGER()},
      {"double", DOUBLE()},   {"boolean", BOOLEAN()}, {"real", REAL()}};
  auto it = kTypes.find(t);
  if (it == kTypes.end()) {
    throw std::invalid_argument("unsupported column type: " + t);
  }
  return it->second;
}

std::vector<std::string> strings(const folly::dynamic& d, const char* key) {
  std::vector<std::string> out;
  if (d.count(key)) {
    for (const auto& v : d[key]) {
      out.push_back(v.asString());
    }
  }
  return out;
}

struct BuildCtx {
  std::shared_ptr<core::PlanNodeIdGenerator> idGen = std::make_shared<core::PlanNodeIdGenerator>();
  memory::MemoryPool* pool;
  int splitsPerFile;
  std::vector<std::pair<core::PlanNodeId, std::vector<std::string>>> scans;
};

core::PlanNodePtr buildPlan(const folly::dynamic& steps, BuildCtx& ctx) {
  exec::test::PlanBuilder b(ctx.idGen, ctx.pool);
  for (const auto& step : steps) {
    const auto op = step["op"].asString();
    if (op == "scan") {
      std::vector<std::string> names;
      std::vector<TypePtr> types;
      for (const auto& c : step["columns"]) {
        names.push_back(c[0].asString());
        types.push_back(parseType(c[1].asString()));
      }
      b.tableScan(ROW(std::move(names), std::move(types)), {},
                  step.getDefault("filter", "").asString());
      ctx.scans.emplace_back(b.planNode()->id(), strings(step, "files"));
    } else if (op == "filter") {
      b.filter(step["expr"].asString());
    } else if (op == "project") {
      b.project(strings(step, "exprs"));
    } else if (op == "agg") {
      auto keys = strings(step, "keys");
      b.partialAggregation(keys, strings(step, "aggs")).localPartition(keys).finalAggregation();
    } else if (op == "join") {
      static const std::unordered_map<std::string, core::JoinType> kJoins = {
          {"inner", core::JoinType::kInner}, {"left", core::JoinType::kLeft},
          {"semi", core::JoinType::kLeftSemiFilter}, {"anti", core::JoinType::kAnti}};
      auto build = buildPlan(step["build"], ctx);
      b.hashJoin(strings(step, "left"), strings(step, "right"), build,
                 step.getDefault("filter", "").asString(), strings(step, "output"),
                 kJoins.at(step["type"].asString()));
    } else if (op == "unnest") {
      std::vector<std::optional<std::string>> names;
      for (auto& n : strings(step, "names")) {
        names.emplace_back(n);
      }
      b.unnest(strings(step, "replicate"), strings(step, "columns"), names);
    } else if (op == "order_by") {
      auto keys = strings(step, "keys");
      b.orderBy(keys, true).localMerge(keys);
    } else if (op == "limit") {
      b.limit(0, step["n"].asInt(), false);
    } else if (op == "write") {
      b.tableWrite(step["dir"].asString(), dwio::common::FileFormat::PARQUET);
    } else {
      throw std::invalid_argument("unknown plan op: " + op);
    }
  }
  return b.planNode();
}

class Result {
 public:
  std::vector<RowVectorPtr> batches;
  std::shared_ptr<memory::MemoryPool> pool;
  std::string stats;
  double seconds{0};

  size_t numBatches() const { return batches.size(); }

  void exportBatch(size_t i, uintptr_t arrayPtr, uintptr_t schemaPtr) const {
    // The Arrow bridge cannot export every encoding (e.g. dictionary-wrapped arrays), so
    // export a flat copy.
    const auto& in = batches.at(i);
    VectorPtr flat = BaseVector::create(in->type(), in->size(), pool.get());
    flat->copy(in.get(), 0, 0, in->size());
    ArrowOptions options;
    options.flattenDictionary = true;
    options.flattenConstant = true;
    exportToArrow(flat, *reinterpret_cast<ArrowSchema*>(schemaPtr), options);
    exportToArrow(flat, *reinterpret_cast<ArrowArray*>(arrayPtr), pool.get(), options);
  }
};

std::shared_ptr<Result> execute(const std::string& planJson, int maxDrivers, bool withStats) {
  init(maxDrivers);
  auto& rt = runtime();
  auto result = std::make_shared<Result>();
  result->pool = rt.outputPool;
  BuildCtx ctx{.pool = result->pool.get()};
  auto plan = buildPlan(folly::parseJson(planJson), ctx);

  const auto start = std::chrono::steady_clock::now();
  {
    py::gil_scoped_release release;
    exec::CursorParameters params;
    params.planNode = plan;
    params.maxDrivers = maxDrivers;
    params.queryCtx = core::QueryCtx::Builder().executor(rt.executor.get()).pool(
        memory::memoryManager()->addRootPool()).build();
    params.outputPool = result->pool;
    auto cursor = exec::TaskCursor::create(params);
    for (auto& [scanId, files] : ctx.scans) {
      for (const auto& file : files) {
        const uint64_t size = std::filesystem::file_size(file);
        const uint64_t n = std::clamp<uint64_t>(size / (4 << 20), 1, 4 * maxDrivers);
        const uint64_t chunk = (size + n - 1) / n;
        for (uint64_t off = 0; off < size; off += chunk) {
          cursor->task()->addSplit(scanId, exec::Split{std::make_shared<connector::hive::HiveConnectorSplit>(
              kConnectorId, file, dwio::common::FileFormat::PARQUET, off, chunk)});
        }
      }
      cursor->task()->noMoreSplits(scanId);
    }
    while (cursor->moveNext()) {
      result->batches.push_back(cursor->current());
    }
    if (withStats) {
      result->stats = exec::printPlanWithStats(*plan, cursor->task()->taskStats(), false);
    }
  }
  result->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return result;
}

} // namespace
} // namespace loopbench

namespace py = pybind11;

PYBIND11_MODULE(lbvelox, m) {
  using loopbench::Result;
  m.doc() = "loopbench Velox extension: text-signature functions and a JSON-plan executor";
  m.def("init", &loopbench::init, py::arg("threads") = 16,
        "Initialise Velox (memory, functions, Parquet, hive connector); idempotent.");
  m.def("execute", &loopbench::execute, py::arg("plan"), py::arg("max_drivers") = 16,
        py::arg("with_stats") = false);
  py::class_<Result, std::shared_ptr<Result>>(m, "Result")
      .def("num_batches", &Result::numBatches)
      .def("export_batch", &Result::exportBatch)
      .def_readonly("stats", &Result::stats)
      .def_readonly("seconds", &Result::seconds);
}
