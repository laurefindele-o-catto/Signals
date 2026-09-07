
#python3 test_ntt.py inputs/4.txt




import os
import sys

from bigmul import multiply, to_limbs, from_limbs
from transforms import NTTTransformer, next_power_of_two


NTT_BASE_DIGITS = 2


def read_operands(path):
    """Return the two operands as decimal STRINGS (multiply() expects text)."""
    nums = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                nums.append(line)
            if len(nums) == 2:
                break
    return nums[0], nums[1]


def multiply_ntt(text_a, text_b):
    """
    Same pipeline as bigmul.multiply_transform, but with the NTT engine:
    limbs -> pad to power of two -> transform -> pointwise mod-p multiply
    -> inverse -> carry sweep. No rounding anywhere -- every value is an
    exact integer mod NTTTransformer.MOD throughout.
    """
    sign_a, limbs_a = to_limbs(text_a, base_digits=NTT_BASE_DIGITS)
    sign_b, limbs_b = to_limbs(text_b, base_digits=NTT_BASE_DIGITS)
    sign = sign_a * sign_b

    engine = NTTTransformer()
    N = next_power_of_two(len(limbs_a) + len(limbs_b) - 1)

    a_padded = [0] * N
    a_padded[:len(limbs_a)] = [int(v) for v in limbs_a]
    b_padded = [0] * N
    b_padded[:len(limbs_b)] = [int(v) for v in limbs_b]

    A = engine.transform(a_padded)
    B = engine.transform(b_padded)
    Y = [(int(x) * int(y)) % engine.MOD for x, y in zip(A, B)]
    result = engine.inverse(Y)   # exact integers, no rounding needed

    return from_limbs(sign, result, base_digits=NTT_BASE_DIGITS), N


def compare_ntt_vs_fft(input_path, out_dir="outputs"):
    text_a, text_b = read_operands(input_path)

    fft_product, fft_N, _, _ = multiply(text_a, text_b, "fft")
    ntt_product, ntt_N = multiply_ntt(text_a, text_b)

    difference = int(fft_product) - int(ntt_product)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ntt_bonus.txt")
    with open(out_path, "w") as f:
        f.write(f"FFT transform N  : {fft_N}\n")
        f.write(f"NTT transform N  : {ntt_N}\n")
        f.write("\n")
        f.write(f"result (fft)     : {fft_product}\n")
        f.write(f"result (ntt)     : {ntt_product}\n")
        f.write(f"difference       : {difference}\n")


if __name__ == "__main__":
    compare_ntt_vs_fft(sys.argv[1])