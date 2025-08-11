//! 軽量な擬似乱数生成器（PCG32）。レイトレーサ用の高速・十分な品質。
//! 参考: O'Neill, "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms".

#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
    inc: u64,
}

impl Rng {
    /// seed と stream を指定して生成（stream は奇数に正規化されます）。
    pub fn new(seed: u64, stream: u64) -> Self {
        let mut rng = Rng { state: 0, inc: (stream << 1) | 1 };
        rng.advance();
        rng.state = rng.state.wrapping_add(seed);
        rng.advance();
        rng
    }

    /// 単一シードから生成（stream は固定）。
    pub fn from_seed(seed: u64) -> Self {
        Self::new(seed, 0x9E3779B97F4A7C15)
    }

    #[inline]
    fn advance(&mut self) {
        // PCG32 LCG ステップ
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(self.inc);
    }

    /// 32bit 乱数。
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let oldstate = self.state;
        self.advance();
        // 出力関数 XSH RR
        let xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) as u32;
        let rot = (oldstate >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// [0,1) の一様乱数（32bit 精度）。
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // 24bit を用いて [0,1) に正規化
        const SCALE: f32 = 1.0 / (1u32 << 24) as f32;
        (self.next_u32() >> 8) as f32 * SCALE
    }

    /// 乱数をスキップ（n ステップ進める）。
    pub fn skip(&mut self, mut n: u64) {
        // 高速なべき乗スキップ（二進展開）。
        let mut cur_mult: u128 = 6364136223846793005u128;
        let mut cur_plus: u128 = self.inc as u128;
        let mut acc_mult: u128 = 1;
        let mut acc_plus: u128 = 0;
        let mut state: u128 = self.state as u128;
        while n > 0 {
            if (n & 1) != 0 {
                acc_mult = acc_mult.wrapping_mul(cur_mult);
                acc_plus = acc_plus.wrapping_mul(cur_mult).wrapping_add(cur_plus);
            }
            cur_plus = cur_plus.wrapping_mul(cur_mult).wrapping_add(cur_plus);
            cur_mult = cur_mult.wrapping_mul(cur_mult);
            n >>= 1;
        }
        state = acc_mult.wrapping_mul(state).wrapping_add(acc_plus);
        self.state = state as u64;
    }
}

/// 64bit の簡易ハッシュ（SplitMix64）。シード拡散に使用。
#[inline]
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}
