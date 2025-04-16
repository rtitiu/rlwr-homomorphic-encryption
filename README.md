# rlwr-homomorphic-encryption
This repo contains a proof of concept implementation of the Homomorphic Encryption scheme based on Ring Learning With Rounding presented in Section 3 of the paper [Designs for practical SHE schemes based on Ring-LWR](https://cic.iacr.org/p/2/1/21/pdf).
The functionality of the scheme (KeyGen, RelinKeyGen, Encrypt, Decrypt, CiphertextAddition, CiphertextMultiplication) is implemented in [LPR_type_he.py](https://github.com/rtitiu/rlwr-homomorphic-encryption/blob/main/LPR_type_he.py) file. It uses the [polymomial multiplication](https://github.com/rtitiu/rlwr-homomorphic-encryption/blob/main/poly_multiplication.py) implemented similar to the [polymul-approx-ffts repo](https://github.com/rtitiu/polymul-approx-ffts). The code in [noise_size.py](https://github.com/rtitiu/rlwr-homomorphic-encryption/blob/main/noise_size.py) file offers a way of measuring the actual noise size that arises in the implementation of the scheme and compares it with the theoretical bounds from the paper.
