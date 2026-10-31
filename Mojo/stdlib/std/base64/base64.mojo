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
"""Provides functions for base64 encoding strings.

You can import these APIs from the `base64` package. For example:

```mojo
from std.base64 import b64encode
```
"""


from ._b64decode import _b64decode
from ._b64encode import _b64encode

# ===-----------------------------------------------------------------------===#
# b64encode
# ===-----------------------------------------------------------------------===#


@inline(.always)
def b64encode(input_bytes: ImmSpan[Byte, _], mut result: String):
    """Performs base64 encoding on the input string.

    Args:
        input_bytes: The input string buffer.
        result: The string in which to store the values.

    Notes:
        This method reserves the necessary capacity. `result` can be a 0
        capacity string.
    """
    _b64encode(input_bytes, result)


@inline(.always)
def b64encode(input_string: StringSlice[_]) -> String:
    """Performs base64 encoding on the input string.

    Args:
        input_string: The input string buffer.

    Returns:
        The ASCII base64 encoded string.
    """
    return b64encode(input_string.as_bytes())


@inline(.always)
def b64encode(input_bytes: ImmSpan[Byte, _]) -> String:
    """Performs base64 encoding on the input string.

    Args:
        input_bytes: The input string buffer.

    Returns:
        The ASCII base64 encoded string.
    """
    var result = String()
    b64encode(input_bytes, result)
    return result^


# ===-----------------------------------------------------------------------===#
# b64decode
# ===-----------------------------------------------------------------------===#


@inline(.always)
def b64decode(str: StringSlice[_]) raises -> List[Byte]:
    """Performs base64 decoding on the input string.

    Whitespace (spaces, tabs, newlines, carriage returns, form feeds, and
    vertical tabs) is ignored, which allows decoding base64 text that has
    been wrapped across multiple lines. Unlike Python's `base64.b64decode`,
    which discards *any* non-alphabet byte, Mojo only ignores whitespace and
    still rejects other invalid characters.

    Args:
        str: A base64 encoded string.

    Returns:
        The decoded bytes.

    Raises:
        If the input length (ignoring whitespace) is not divisible by 4, or
        the input contains a non-whitespace character outside the base64
        alphabet.
    """
    return _b64decode(str.as_bytes())


# ===-----------------------------------------------------------------------===#
# b16encode
# ===-----------------------------------------------------------------------===#


def b16encode(str: StringSlice[_]) -> String:
    """Performs base16 encoding on the input string slice.

    Args:
        str: The input string slice.

    Returns:
        Base16 encoding of the input string.
    """
    comptime lookup = "0123456789ABCDEF"
    var b16chars = lookup.ptr()

    var data = str.as_bytes()
    var length = str.byte_length()
    var result = String(capacity_bytes=length * 2)

    for i in range(length):
        var str_byte = data[i]
        var hi = str_byte >> 4
        var lo = str_byte & 0b1111
        result._unsafe_append_byte(b16chars[unsafe_offset=hi])
        result._unsafe_append_byte(b16chars[unsafe_offset=lo])

    return result^


# ===-----------------------------------------------------------------------===#
# b16decode
# ===-----------------------------------------------------------------------===#


def b16decode(str: StringSlice[_]) raises -> List[Byte]:
    """Performs base16 decoding on the input string.

    Args:
        str: A base16 encoded string.

    Returns:
        The decoded bytes.

    Raises:
        If the input length is odd or any character is outside the base16
        alphabet `[0-9A-F]` (per RFC 4648 section 8, which is strictly
        uppercase).
    """

    comptime `A` = Byte(ord("A"))
    comptime `F` = Byte(ord("F"))
    comptime `0` = Byte(ord("0"))
    comptime `9` = Byte(ord("9"))

    # TODO: Measure perf against lookup table approach
    @inline(.always)
    def decode(c: Byte) raises -> Byte:
        if `0` <= c <= `9`:
            return c - `0`
        elif `A` <= c <= `F`:
            return c - `A` + Byte(10)
        else:
            raise Error(
                "ValueError: Unexpected character '",
                chr(Int(c)),
                "' encountered",
            )

    var data = str.as_bytes()
    var n = str.byte_length()
    if n % 2 != 0:
        raise Error("ValueError: Input length '", n, "' must be divisible by 2")

    var result = List[Byte](capacity=n // 2)

    for i in range(0, n, 2):
        var hi = data[i]
        var lo = data[i + 1]
        result.append(decode(hi) << 4 | decode(lo))

    return result^
