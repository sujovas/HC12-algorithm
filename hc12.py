import numpy as np
import time

# from sympy.combinatorics.graycode import gray_to_bin


class HC12:
    def __init__(self, n_param, n_bit_param, dod_param=None, float_type=np.float64):
        self.n_param = n_param
        self.n_bit_param = np.array([n_bit_param for _ in range(n_param)], dtype=np.uint32)
        self.dod_param = dod_param
        self.uint_type = np.uint16
        self.float_type = float_type
        self.total_bits = int(np.sum(self.n_bit_param))
        self._M0_rows = 1
        self._M1_rows = self.total_bits
        self._M2_rows = self.total_bits * (self.total_bits - 1) // 2
        self.rows = self._M0_rows + self._M1_rows + self._M2_rows
        self.K = np.zeros((1, self.n_param), dtype=self.uint_type)
        self.M = np.zeros((self.rows, self.n_param), dtype=self.uint_type)
        self.B = np.zeros((self.rows, self.n_param), dtype=self.uint_type)
        self.I = np.zeros((self.rows, self.n_param), dtype=self.uint_type)
        self.R = np.zeros((self.rows, self.n_param), dtype=self.float_type)
        self.F = np.zeros(self.rows, dtype=self.float_type)
        self._init_M()
        self.P = []

    # @property
    # def dod_param(self):
    #     return self._dod_param
    #
    # @dod_param.setter
    # def dod_param(self, dod_param):
    #     if len(dod_param) == self.n_param:
    #         self._dod_param = np.array(dod_param)
    #     else:
    #         self._dod_param = np.array([dod_param for _ in range(self.n_param)])

    def run(self, func, times, max_iter=10000):
        # dod = self._dod_param
        global best_idx, iter_i
        dod = []
        for i in range(self.n_param):
            dod.append(self.dod_param)
        n_bit = self.n_bit_param
        x_out = np.zeros((times, self.n_param), dtype=self.float_type)
        fval = np.full(times, float('inf'))
        # print(dod)

        def interval_to_float(int_i, a, b, n_bits):
            return (b - a) / (2 ** n_bits - 1) * int_i + a

        iterations = np.zeros((times, 1))
        winning_run = 0

        for run_i in range(times):
            start = time.process_time()
            # nachystat K
            self.K[:] = [np.random.randint(0, n_bit[i] ** 2 - 1) for i in range(self.n_param)]
            run_fval = float('inf')
            for iter_i in range(max_iter):
                # K xor M - vysledek B
                np.bitwise_xor(self.K, self.M, out=self.B)
                # dekodovat Graye z B do I
                np.bitwise_and(self.B, 1 << n_bit, out=self.I)
                for par in range(self.n_param):
                    for bit in range(n_bit[par], 0, -1):
                        self.I[:, par] |= np.bitwise_xor((self.I[:, par] & 1 << bit) >> 1, self.B[:, par] & 1 << (bit - 1))
            # prevod I do realnych cisel
                    self.R[:, par] = interval_to_float(self.I[:, par], dod[par][0], dod[par][1], n_bit[par])
            #         self.R[:,par] = interval_to_float(self.I[:,par], dod[par, 0], dod[par,1], n_bit[par])
            # vypocet hodnoty ucelove funkce F
                func(self.R, out=self.F)
                # print('self.R', self.R, 'self.F', self.F)
            # vybrat nejlepsi a pak bud ukoncit nebo ho prohlasit za nove K
                best_idx = np.argmin(self.F)
                min_value = np.min(self.F)
                self.P.append(min_value)
                # print('P', self.P)
                run_fval = self.F[best_idx]
                if best_idx == 0:
                    break
                self.K = self.B[best_idx,:]
            iterations[run_i] = iter_i
            x_out[run_i, :] = self.R[best_idx, :]
            fval[run_i] = run_fval
            if run_fval <min(fval):
                winning_run = run_i
        # print('winning run', winning_run, 'iteration', iterations[winning_run], 'avg number of iterations', np.mean(iterations))
        return x_out, fval


    def _init_M(self):
        bit_lookup = []
        for p in range(self.n_param):
            for b in range(self.n_bit_param[p]):
                bit_lookup.append((p, b))

        for j in range(1, 1 + self._M1_rows):
            p, bit = bit_lookup[j - 1]
            self.M[j, p] |= 1 << bit

        j = self._M0_rows + self._M1_rows
        for bit in range(self.total_bits - 1):
            for bit2 in range(bit + 1, self.total_bits):
                self.M[j, bit_lookup[bit][0]] |= 1 << bit_lookup[bit][1]
                self.M[j, bit_lookup[bit2][0]] |= 1 << bit_lookup[bit2][1]
                j += 1
