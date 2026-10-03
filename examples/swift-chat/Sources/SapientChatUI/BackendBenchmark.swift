import Foundation
import Sapient

/// CPU-versus-GPU benchmark for a real device, driven by a launch argument:
///
///     -benchmark <model-alias> [-benchmark-rounds N] [-benchmark-tokens N]
///                              [-benchmark-backends cpu,wgpu]
///
/// It loads the model on each backend in turn (`cpu`, then `wgpu`, repeated
/// for N rounds so thermal drift shows), runs `LlmSession.benchmark`, and
/// prints one `SAPIENT_BENCH {json}` line per measurement plus a summary.
/// Lines go to stdout, so they appear in Xcode's console and in
/// `xcrun devicectl device process launch --console`. See
/// docs/MOBILE.md "CPU versus GPU on a real iPhone".
public enum BackendBenchmark {
    /// One finished measurement, or the reason a backend could not be measured.
    public struct Row {
        public let backend: String
        public let round: Int
        public let report: BenchmarkReport?
        public let error: String?
    }

    /// Parsed launch arguments, or `nil` when `-benchmark` is absent.
    public struct Request {
        public let model: String
        public let rounds: Int
        public let maxTokens: UInt32
        /// Backends to measure, in order. `peak_mb` is the process's high-water
        /// mark, so measure one backend per launch when memory is the question.
        public let backends: [String]

        public static func fromLaunchArguments(
            _ args: [String] = ProcessInfo.processInfo.arguments
        ) -> Request? {
            func value(_ flag: String) -> String? {
                guard let i = args.firstIndex(of: flag), args.indices.contains(i + 1) else {
                    return nil
                }
                return args[i + 1]
            }
            guard let model = value("-benchmark") else { return nil }
            return Request(
                model: model,
                rounds: max(1, Int(value("-benchmark-rounds") ?? "") ?? 2),
                maxTokens: UInt32(value("-benchmark-tokens") ?? "") ?? 128,
                backends: (value("-benchmark-backends") ?? "cpu,wgpu")
                    .split(separator: ",").map { String($0) }
            )
        }
    }

    /// Runs the whole comparison on the calling thread (call it off the main
    /// thread: every step blocks). `progress` receives each line as printed.
    public static func run(_ request: Request, progress: @escaping (String) -> Void) -> [Row] {
        func emit(_ line: String) {
            print(line)
            fflush(stdout) // piped stdout is block-buffered; a log reader must see each line
            progress(line)
        }
        emit("SAPIENT_BENCH_START model=\(request.model) rounds=\(request.rounds) "
            + "tokens=\(request.maxTokens) backends=\(request.backends.joined(separator: ",")) "
            + "sapient=\(version())")
        var rows: [Row] = []
        for round in 1...request.rounds {
            for backend in request.backends {
                emit("… round \(round)/\(request.rounds): \(backend) (loading)")
                do {
                    // A fresh session per measurement: the previous one is
                    // released first, so two copies of the model never coexist.
                    let session = try LlmSession.load(
                        model: request.model,
                        options: GenerationOptions(maxTokens: request.maxTokens, backend: backend)
                    )
                    let report = try session.benchmark(
                        options: BenchmarkOptions(maxTokens: request.maxTokens, runs: 3, warmup: 1),
                        listener: nil
                    )
                    rows.append(Row(backend: backend, round: round, report: report, error: nil))
                    emit("SAPIENT_BENCH " + json(backend: backend, round: round, report: report))
                } catch {
                    rows.append(Row(backend: backend, round: round, report: nil, error: "\(error)"))
                    emit("SAPIENT_BENCH_ERROR backend=\(backend) round=\(round) \(error)")
                }
            }
        }
        emit(summary(rows))
        emit("SAPIENT_BENCH_DONE")
        return rows
    }

    static func json(backend: String, round: Int, report r: BenchmarkReport) -> String {
        let peak = r.peakFootprintBytes.map { String($0 / 1_048_576) } ?? "null"
        let eos = r.runs.filter { $0.hitEos }.count
        return "{\"backend\":\"\(backend)\",\"round\":\(round),"
            + "\"resolved\":\"\(r.backendLabel)\","
            + "\"decode_tps\":\(String(format: "%.2f", r.meanDecodeTokensPerSec)),"
            + "\"decode_tps_min\":\(String(format: "%.2f", r.minDecodeTokensPerSec)),"
            + "\"decode_tps_max\":\(String(format: "%.2f", r.maxDecodeTokensPerSec)),"
            + "\"prefill_tps\":\(String(format: "%.1f", r.meanPrefillTokensPerSec)),"
            + "\"ttft_ms\":\(r.meanTtftMs),\"load_ms\":\(r.loadTimeMs),"
            + "\"peak_mb\":\(peak),\"runs_hit_eos\":\(eos),"
            + "\"thermal_start\":\"\(r.thermalStart)\",\"thermal_end\":\"\(r.thermalEnd)\"}"
    }

    static func summary(_ rows: [Row]) -> String {
        func mean(_ backend: String) -> Double? {
            let v = rows.filter { $0.backend == backend }.compactMap { $0.report?.meanDecodeTokensPerSec }
            return v.isEmpty ? nil : v.reduce(0, +) / Double(v.count)
        }
        guard let cpu = mean("cpu"), let gpu = mean("wgpu") else {
            return "SAPIENT_BENCH_SUMMARY incomplete: one backend has no measurement"
        }
        let faster = gpu > cpu ? "GPU" : "CPU"
        return String(
            format: "SAPIENT_BENCH_SUMMARY cpu %.1f tok/s · gpu %.1f tok/s · gpu/cpu %.2fx · faster: %@",
            cpu, gpu, gpu / cpu, faster
        )
    }
}
