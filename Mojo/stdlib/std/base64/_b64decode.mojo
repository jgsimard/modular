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
"""SIMD helpers for Base64 decoding.

The nibble lookup tables and packing algorithm follow Muła and Lemire,
"Faster Base64 Encoding and Decoding Using AVX2 Instructions", section 4.2:
https://arxiv.org/abs/1704.00605.
"""

from std.memory import bitcast
from std.sys import Endian, simd_byte_width
from std.utils import IndexList


def _count_base64_chars(input: ImmSpan[Byte, _]) -> Int:
    """Counts non-whitespace bytes without validating the alphabet."""
    comptime width = simd_byte_width()
    var pos = 0
    var count = 0
    while len(input) - pos >= width:
        var chars = (
            input.unsafe_ptr().unsafe_offset(pos).unsafe_load[width=width]()
        )
        var whitespace = chars.eq(32) | (chars.ge(9) & chars.le(13))
        count += width - whitespace.reduce_bit_count()
        pos += width
    while pos < len(input):
        var char = input[pos]
        if not (char == 32 or (char >= 9 and char <= 13)):
            count += 1
        pos += 1
    return count


@inline(.always)
def _decode_base64_block[
    width: SIMDLength
](chars: SIMD[.uint8, width]) -> Tuple[SIMD[.uint8, width], Bool]:
    """Decodes an alphabet-only block, returning packed bytes and success.

    Only the first `3 * width // 4` output bytes are significant. Whitespace,
    padding, and invalid bytes return False so the caller can handle them in
    input order with the scalar decoder.
    """
    comptime assert width >= 4, "Base64 blocks must contain whole quartets"
    # A byte is in the alphabet iff the intersection of these two bitsets
    # is empty. All lookup indices are in [0, 15], including non-ASCII input.
    comptime low_bits = SIMD[.uint8, 16](
        21, 17, 17, 17, 17, 17, 17, 17, 17, 17, 19, 26, 27, 27, 27, 26
    )
    comptime high_bits = SIMD[.uint8, 16](
        16, 16, 1, 2, 4, 8, 4, 8, 16, 16, 16, 16, 16, 16, 16, 16
    )
    comptime offsets = SIMD[.uint8, 16](
        0, 16, 19, 4, -65, -65, -71, -71, 0, 0, 0, 0, 0, 0, 0, 0
    )
    var hi = chars >> 4
    var invalid = low_bits._dynamic_shuffle(
        chars & 15
    ) & high_bits._dynamic_shuffle(hi)
    if any(invalid.ne(0)):
        return chars, False

    # '/' uses offset 16, whereas '+' uses offset 19.
    var indices = hi - chars.eq(47).select(1, 0).cast[.uint8]()
    var values = chars + offsets._dynamic_shuffle(indices)

    # Merge adjacent 6-bit values into 12-bit pairs, then 24-bit quartets.
    var pairs = bitcast[.uint16, width // 2](values)
    comptime if Endian.native() == .little:
        pairs = ((pairs & 0x3F) << 6) | (pairs >> 8)
    else:
        pairs = ((pairs >> 8) << 6) | (pairs & 0x3F)
    var quads = bitcast[.uint32, width // 4](pairs)
    comptime if Endian.native() == .little:
        quads = ((quads & 0xFFF) << 12) | (quads >> 16)
    else:
        quads = ((quads >> 16) << 12) | (quads & 0xFFF)

    def packing_mask() -> IndexList[width]:
        var mask = IndexList[width]()
        for i in range(3 * width // 4):
            comptime if Endian.native() == .little:
                mask[i] = 4 * (i // 3) + 2 - i % 3
            else:
                mask[i] = 4 * (i // 3) + 1 + i % 3
        return mask

    return bitcast[.uint8, width](quads).shuffle[mask=packing_mask()](), True
