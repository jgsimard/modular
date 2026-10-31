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

"""
We make use of the following papers for the implementation, note that there
are some small differences.

Wojciech Muła, Daniel Lemire, Base64 encoding and decoding at almost the
speed of a memory copy, Software: Practice and Experience 50 (2), 2020.
https://arxiv.org/abs/1910.05109

Wojciech Muła, Daniel Lemire, Faster Base64 Encoding and Decoding using AVX2
Instructions, ACM Transactions on the Web 12 (3), 2018.
https://arxiv.org/abs/1704.00605
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


@inline(.never)
def _decode_base64_run(
    input: ImmSpan[Byte, _], output: MutSpan[Byte, _]
) -> Int:
    """Decodes consecutive alphabet-only SIMD blocks into `output`.

    Keep this loop separate from scalar fallback so the compiler can hoist its
    lookup tables and keep them in registers across blocks.

    Args:
        input: The encoded bytes.
        output: Space for at least `3 * (len(input) // 4)` decoded bytes.

    Returns:
        The number of input bytes consumed, stopping before the first block
        containing whitespace, padding, or an invalid byte, or a partial block.
    """
    comptime width = simd_byte_width()
    var pos = 0
    var out_pos = 0
    while len(input) - pos >= width:
        var chars = (
            input.unsafe_ptr().unsafe_offset(pos).unsafe_load[width=width]()
        )
        var decoded, valid = _decode_base64_block(chars)
        if not valid:
            break
        var decoded_bytes = decoded.as_bytes()
        output[out_pos : out_pos + 3 * width // 4].copy_from(
            Span(decoded_bytes)[: 3 * width // 4]
        )
        pos += width
        out_pos += 3 * width // 4
    return pos


@inline(.always)
def _ascii_to_value(char: Byte) raises -> Byte:
    """Converts an ASCII character to its integer value for base64 decoding.

    Args:
        char: A single ascii byte.

    Returns:
        The integer value of the character for base64 decoding.

    Raises:
        If the character is outside the base64 alphabet.
    """
    comptime `A` = Byte(ord("A"))
    comptime `a` = Byte(ord("a"))
    comptime `Z` = Byte(ord("Z"))
    comptime `z` = Byte(ord("z"))
    comptime `0` = Byte(ord("0"))
    comptime `9` = Byte(ord("9"))
    comptime `=` = Byte(ord("="))
    comptime `+` = Byte(ord("+"))
    comptime `/` = Byte(ord("/"))

    # TODO: Measure perf against lookup table approach
    if char == `=`:
        return Byte(0)
    elif `A` <= char <= `Z`:
        return char - `A`
    elif `a` <= char <= `z`:
        return char - `a` + Byte(26)
    elif `0` <= char <= `9`:
        return char - `0` + Byte(52)
    elif char == `+`:
        return Byte(62)
    elif char == `/`:
        return Byte(63)
    else:
        raise Error(
            "ValueError: Unexpected character '",
            chr(Int(char)),
            "' encountered",
        )


@inline(.always)
def _is_ascii_whitespace(char: Byte) -> Bool:
    """Returns True if `char` is one of the six ASCII whitespace bytes.

    `Codepoint.is_posix_space()` is not usable here: it also accepts the file,
    group and record separators (0x1C-0x1E), so it would let those bytes
    through the base64 alphabet check.

    Args:
        char: A single byte.

    Returns:
        True if the byte is a space, tab, newline, vertical tab, form feed, or
        carriage return.
    """
    comptime ` ` = Byte(ord(" "))
    comptime `\t` = Byte(ord("\t"))
    comptime `\r` = Byte(ord("\r"))

    # \t, \n, \v, \f and \r occupy the contiguous range 0x09-0x0D.
    return char == ` ` or (char >= `\t` and char <= `\r`)


@inline(.always)
def _next_significant_byte(
    data: ImmSpan[Byte, _], pos: Int
) -> Tuple[Byte, Int]:
    """Returns the next non-whitespace byte in `data` and the position past it.

    The caller must guarantee that a non-whitespace byte remains at or after
    `pos`; this is not checked here.

    Args:
        data: The bytes to scan.
        pos: The position to start scanning from.

    Returns:
        The next non-whitespace byte, and the position one past that byte.
    """
    var i = pos
    while _is_ascii_whitespace(data[i]):
        i += 1
    return data[i], i + 1


@inline(.never)
def _b64decode(input: ImmSpan[Byte, _]) raises -> List[Byte]:
    """Decodes Base64 bytes, ignoring ASCII whitespace.

    Args:
        input: The encoded bytes.

    Returns:
        The decoded bytes.

    Raises:
        If the significant input length is not divisible by 4, or an invalid
        character is encountered before padding terminates decoding.
    """
    comptime `=` = Byte(ord("="))

    var n = _count_base64_chars(input)

    if n % 4 != 0:
        raise Error(
            t"ValueError: Input length '{n}' (ignoring whitespace) must be"
            t" divisible by 4"
        )

    var result = List[Byte](unsafe_uninit_length=(n // 4) * 3)

    # This algorithm is based on https://arxiv.org/abs/1704.00605. It reads
    # directly from `input`, skipping over whitespace as it goes, so it never
    # allocates or copies a whitespace-free version of the input.
    comptime width = simd_byte_width()
    var pos = 0
    var remaining = n
    var output_pos = 0
    # The counting pass also identifies inputs that can use a tight SIMD loop
    # without repeatedly checking for whitespace between blocks.
    if n == len(input) and n >= width:
        pos = _decode_base64_run(input, result)
        remaining -= pos
        output_pos = 3 * pos // 4
    while remaining > 0:
        # Avoid repeatedly probing whitespace-heavy input with the SIMD path.
        if len(input) - pos >= width and not _is_ascii_whitespace(input[pos]):
            var chars = (
                input.unsafe_ptr().unsafe_offset(pos).unsafe_load[width=width]()
            )
            var decoded, valid = _decode_base64_block(chars)
            if valid:
                # Store only the packed prefix, never past the output buffer.
                var decoded_bytes = decoded.as_bytes()
                result[output_pos : output_pos + 3 * width // 4].copy_from(
                    decoded_bytes[: 3 * width // 4]
                )
                pos += width
                remaining -= width
                output_pos += 3 * width // 4
                continue

        # Consume one significant quartet before retrying SIMD. This preserves
        # whitespace handling, padding termination, and scalar error ordering.
        var a_byte, a_pos = _next_significant_byte(input, pos)
        var b_byte, b_pos = _next_significant_byte(input, a_pos)
        var c_byte, c_pos = _next_significant_byte(input, b_pos)
        var d_byte, d_pos = _next_significant_byte(input, c_pos)
        pos = d_pos
        remaining -= 4

        var a = _ascii_to_value(a_byte)
        var b = _ascii_to_value(b_byte)
        var c = _ascii_to_value(c_byte)
        var d = _ascii_to_value(d_byte)

        result[output_pos] = (a << 2) | (b >> 4)
        output_pos += 1
        if c_byte == `=`:
            break
        result[output_pos] = ((b & 0x0F) << 4) | (c >> 2)
        output_pos += 1
        if d_byte == `=`:
            break
        result[output_pos] = ((c & 0x03) << 6) | d
        output_pos += 1

    result.shrink(output_pos)
    return result^
