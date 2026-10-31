# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Benchmarks Base64 decoding, including allocation and whitespace handling."""

from std.base64 import b64decode, b64encode
from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
    black_box,
    keep,
)


def bench_decode(mut b: Bencher, input: String) raises:
    @inline(.always)
    def decode() raises {imm}:
        var result = b64decode(StringSlice(black_box(input)))
        keep(result)

    b.iter(decode)


def main() raises:
    var bench = Bench(BenchConfig(num_repetitions=5))
    # Source-byte lengths cover padding, SIMD boundaries, and larger buffers.
    var sizes: List[Int] = [0, 1, 2, 3, 12, 24, 48, 96, 1024, 65536, 1048576]
    var spacings: List[Int] = [0, 76, 1]
    for size in sizes:
        var data = List[Byte](capacity=size)
        for i in range(size):
            data.append(Byte((73 * i + 19) % 256))
        var encoded = b64encode(data)
        for spacing in spacings:
            var input = String()
            if spacing == 0:
                input = encoded.copy()
            else:
                for pos in range(0, encoded.byte_length(), spacing):
                    input += encoded[
                        byte = pos : min(pos + spacing, encoded.byte_length())
                    ]
                    input += "\r\n"
            bench.bench_with_input(
                bench_decode,
                BenchId("b64decode", String(t"bytes={size}/spacing={spacing}")),
                input,
                # Report decoded bytes/s so whitespace doesn't inflate throughput.
                [ThroughputMeasure(BenchMetric.bytes, size)],
            )
    print(bench)
