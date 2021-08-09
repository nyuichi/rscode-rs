use proptest::prelude::*;
use proptest_derive::Arbitrary;

// Galois field. The internal value is vector representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Arbitrary)]
struct Gf<T>(pub T);

//     | 0x0 0x1 0x2 0x3 0x4 0x5 0x6 0x7 0x8 0x9 0xa 0xb 0xc 0xd 0xe 0xf
// ----+----------------------------------------------------------------
// 0x0 | 001 002 004 008 016 032 064 128 029 058 116 232 205 135 019 038
// 0x1 | 076 152 045 090 180 117 234 201 143 003 006 012 024 048 096 192
// 0x2 | 157 039 078 156 037 074 148 053 106 212 181 119 238 193 159 035
// 0x3 | 070 140 005 010 020 040 080 160 093 186 105 210 185 111 222 161
// 0x4 | 095 190 097 194 153 047 094 188 101 202 137 015 030 060 120 240
// 0x5 | 253 231 211 187 107 214 177 127 254 225 223 163 091 182 113 226
// 0x6 | 217 175 067 134 017 034 068 136 013 026 052 104 208 189 103 206
// 0x7 | 129 031 062 124 248 237 199 147 059 118 236 197 151 051 102 204
// 0x8 | 133 023 046 092 184 109 218 169 079 158 033 066 132 021 042 084
// 0x9 | 168 077 154 041 082 164 085 170 073 146 057 114 228 213 183 115
// 0xa | 230 209 191 099 198 145 063 126 252 229 215 179 123 246 241 255
// 0xb | 227 219 171 075 150 049 098 196 149 055 110 220 165 087 174 065
// 0xc | 130 025 050 100 200 141 007 014 028 056 112 224 221 167 083 166
// 0xd | 081 162 089 178 121 242 249 239 195 155 043 086 172 069 138 009
// 0xe | 018 036 072 144 061 122 244 245 247 243 251 235 203 139 011 022
// 0xf | 044 088 176 125 250 233 207 131 027 054 108 216 173 071 142 001

const GF256_REG_TO_VEC: [u8; 256] =
    array_const_fn_init::array_const_fn_init![gf256_reg_to_vec; 256];

// GF(2**8) = GF(256)
// is given by extending GF(2) with a such that p(a) = 0, where p(x) = x**8 + x**4 + x**3 + x**2 + 1.
const PRIM: u8 = 0x1d;

const fn gf256_reg_to_vec(i: usize) -> u8 {
    let mut n = 1;
    let mut c = 0;
    while c != i {
        // (n[7] * a**7 + n[6] * a**6 + ... + n[1] * a + n[0]) * a
        //   = n[7] * a**8 + n[6] * a**6 + ... + n[1] * a**2 + n[0] * a
        //   = n[7] * (a**4 + a**3 + a**2 + 1) + n[6] * a**6 +  ... + n[1] * a**2 + n[0] * a
        n = (n << 1) ^ ((n >> 7) * PRIM);
        c += 1;
    }
    n
}

#[allow(unused)]
fn print_exp_table() {
    print!("    | ");
    for i in 0..16 {
        print!("{:#x} ", i);
    }
    println!();
    println!("----+{}", "-".repeat(64));
    for i in 0..16 {
        print!("{:#x} | ", i);
        for j in 0..16 {
            print!("{:#03} ", Gf::<u8>::exp(i * 16 + j).0);
        }
        println!();
    }
}

const GF256_VEC_TO_REG: [u8; 256] =
    array_const_fn_init::array_const_fn_init![gf256_vec_to_reg; 256];

const fn gf256_vec_to_reg(n: usize) -> u8 {
    if n == 0 {
        255 // dummy
    } else {
        let mut i: u8 = 0;
        while GF256_REG_TO_VEC[i as usize] != (n as u8) {
            i = i.wrapping_add(1);
        }
        i
    }
}

/// Gf<u8> is GF(2**8) = GF(256)
impl Gf<u8> {
    fn exp(i: usize) -> Self {
        unsafe { Self(*GF256_REG_TO_VEC.get_unchecked(i % 255)) }
    }
    fn log(self) -> usize {
        if self.0 == 0 {
            panic!("divide by zero");
        }
        unsafe { *GF256_VEC_TO_REG.get_unchecked(self.0 as usize) as usize }
    }
    fn inv(self) -> Self {
        if self.0 == 0 {
            panic!("divide by zero");
        }
        Self::exp(255 - self.log())
    }
    fn pow(self, i: usize) -> Self {
        if self.0 == 0 {
            self
        } else {
            Self::exp(self.log() * i)
        }
    }
}

#[test]
fn exp_0() {
    assert_eq!(Gf(1u8), Gf::exp(0));
}

proptest! {
    #[test]
    fn exp_log_roundtrip(i in 0..256usize) {
        let j = Gf::<u8>::exp(i).log();
        if i == 255 {
            prop_assert_eq!(j, 0);
        } else {
            prop_assert_eq!(j, i);
        }
    }

    #[test]
    fn log_exp_roundtrip(n: Gf<u8>) {
        if n.0 != 0 {
            prop_assert_eq!(Gf::<u8>::exp(n.log()), n);
        }
    }
}

impl std::ops::Add for Gf<u8> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        #[allow(clippy::suspicious_arithmetic_impl)]
        Self(self.0 ^ rhs.0)
    }
}

impl std::ops::AddAssign for Gf<u8> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl std::ops::Sub for Gf<u8> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        #[allow(clippy::suspicious_arithmetic_impl)]
        Self(self.0 ^ rhs.0)
    }
}

impl std::ops::SubAssign for Gf<u8> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl std::ops::Mul for Gf<u8> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.0 == 0 || rhs.0 == 0 {
            Self(0)
        } else {
            Self::exp(self.log() + rhs.log())
        }
    }
}

proptest! {
    #[test]
    fn mul_assoc(n: Gf<u8>, m: Gf<u8>, l: Gf<u8>) {
        prop_assert_eq!((n * m) * l, n * (m * l));
    }

    #[test]
    fn mul_right_inv(n: Gf<u8>) {
        if n.0 != 0 {
            prop_assert_eq!(Gf(1u8), n * n.inv());
        }
    }

    #[test]
    fn add_mul_left_distr(n: Gf<u8>, m: Gf<u8>, l: Gf<u8>) {
        prop_assert_eq!((n + m) * l, n * l + m * l);
    }

    #[test]
    fn pow_mul(n: Gf<u8>, i in 0..256usize) {
        let mut m = Gf(1u8);
        for _ in 0..i {
            m *= n;
        }
        prop_assert_eq!(n.pow(i), m);
    }
}

impl std::ops::MulAssign for Gf<u8> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl std::ops::Div for Gf<u8> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        if rhs.0 == 0 {
            panic!("divide by zero");
        }
        if self.0 == 0 || rhs.0 == 1 {
            self
        } else {
            Self::exp(self.log() + 255 - rhs.log())
        }
    }
}

proptest! {
    #[test]
    fn div_by_inv(n: Gf<u8>, m: Gf<u8>) {
        if m.0 != 0 {
            prop_assert_eq!(n / m, n * m.inv());
        }
    }
}

impl std::ops::DivAssign for Gf<u8> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

// The inner Vec must not have leading zeros.
#[derive(Clone, Debug, PartialEq, Eq, Arbitrary)]
struct Poly<T>(pub Vec<T>);

#[cfg(test)]
fn poly(size: impl Into<proptest::collection::SizeRange>) -> impl Strategy<Value = Poly<Gf<u8>>> {
    proptest::collection::vec(any::<Gf<u8>>(), size)
        .prop_filter("leading zeros", |v| {
            v.last().map(|n| n.0 != 0).unwrap_or(true)
        })
        .prop_map(Poly)
}

fn yank_leading_zeros(v: &mut Vec<Gf<u8>>) {
    while let Some(n) = v.last() {
        if n.0 != 0 {
            break;
        }
        v.pop();
    }
}

impl std::ops::AddAssign<&Self> for Poly<Gf<u8>> {
    fn add_assign(&mut self, rhs: &Self) {
        if self.0.len() >= rhs.0.len() {
            for i in 0..rhs.0.len() {
                self.0[i] += rhs.0[i];
            }
        } else {
            for i in 0..self.0.len() {
                self.0[i] += rhs.0[i];
            }
            self.0.extend(&rhs.0[self.0.len()..]);
        }
        yank_leading_zeros(&mut self.0);
    }
}

#[cfg(test)]
impl std::ops::Add for Poly<Gf<u8>> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut x = self;
        x += &rhs;
        x
    }
}

#[cfg(test)]
impl std::ops::Add<&Self> for Poly<Gf<u8>> {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        let mut x = self;
        x += rhs;
        x
    }
}

impl std::ops::MulAssign<&Self> for Poly<Gf<u8>> {
    fn mul_assign(&mut self, rhs: &Self) {
        if rhs.0.is_empty() {
            self.0.clear();
            return;
        }
        if self.0.is_empty() {
            return;
        }
        let mut z = vec![Gf(0u8); self.0.len() + rhs.0.len() - 1];
        for (i, n) in self.0.iter().copied().enumerate() {
            for (j, m) in rhs.0.iter().copied().enumerate() {
                z[i + j] += n * m;
            }
        }
        yank_leading_zeros(&mut z);
        self.0 = z;
    }
}

impl std::ops::Mul for Poly<Gf<u8>> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut x = self;
        x *= &rhs;
        x
    }
}

#[cfg(test)]
impl std::ops::Mul<&Self> for Poly<Gf<u8>> {
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self::Output {
        let mut x = self;
        x *= rhs;
        x
    }
}

impl std::ops::RemAssign<&Self> for Poly<Gf<u8>> {
    fn rem_assign(&mut self, rhs: &Self) {
        if rhs.0.is_empty() {
            panic!("divide by zero");
        }
        if self.0.len() < rhs.0.len() {
            return;
        }
        for _ in 0..self.0.len() - rhs.0.len() + 1 {
            let n = self.0[self.0.len() - 1] / rhs.0[rhs.0.len() - 1];
            for (x, y) in self.0.iter_mut().rev().zip(rhs.0.iter().rev()) {
                *x -= n * *y;
            }
            self.0.pop();
        }
        yank_leading_zeros(&mut self.0);
    }
}

proptest! {
    #[test]
    fn test_rem_assign(x in poly(0..10), y in poly(1..10), z in poly(0..10)) {
        let mut r = z.clone();
        r %= &y;
        let mut w = x * y.clone() + z;
        w %= &y;
        prop_assert_eq!(r, w);
    }
}

impl Poly<Gf<u8>> {
    fn div(&mut self, rhs: &Self) -> Self {
        if rhs.0.is_empty() {
            panic!("divide by zero");
        }
        if self.0.len() < rhs.0.len() {
            return Self(vec![]);
        }
        let mut denom = vec![Gf(0u8); self.0.len() - rhs.0.len() + 1];
        for n in denom.iter_mut().rev() {
            *n = self.0[self.0.len() - 1] / rhs.0[rhs.0.len() - 1];
            for (x, y) in self.0.iter_mut().rev().zip(rhs.0.iter().rev()) {
                *x -= *n * *y;
            }
            self.0.pop();
        }
        yank_leading_zeros(&mut self.0);
        Self(denom)
    }

    // differentiation
    fn diff(&mut self) {
        self.0.remove(0);
        for i in 0..self.0.len() {
            if i % 2 == 1 {
                self.0[i] = Gf(0);
            }
        }
    }

    fn apply(&self, n: Gf<u8>) -> Gf<u8> {
        let mut r = Gf(0);
        for (i, m) in self.0.iter().copied().enumerate() {
            r += m * n.pow(i);
        }
        r
    }

    #[cfg(test)]
    fn degree(&self) -> usize {
        match self.0.len() {
            0 | 1 => 0,
            n => n - 1,
        }
    }

    fn scale(&mut self, k: Gf<u8>) {
        if k.0 == 0 {
            self.0.clear();
        } else {
            for n in self.0.iter_mut() {
                *n *= k;
            }
        }
    }

    #[cfg(test)]
    fn gcd(&self, rhs: &Self) -> Self {
        let mut r0 = self.clone();
        let mut r1 = rhs.clone();
        if r0.0.len() < r1.0.len() {
            std::mem::swap(&mut r0, &mut r1);
        }
        while !r1.0.is_empty() {
            r0 %= &r1;
            std::mem::swap(&mut r0, &mut r1);
        }
        if !r0.0.is_empty() {
            r0.scale(r0.0.last().unwrap().inv());
        }
        r0
    }
}

proptest! {
    #[test]
    fn poly_mul_assoc(x in poly(0..10), y in poly(0..10), z in poly(0..10)) {
        prop_assert_eq!((x.clone() * y.clone()) * z.clone(), x * (y * z));
    }

    #[test]
    fn poly_left_distr(x in poly(0..10), y in poly(0..10), z in poly(0..10)) {
        prop_assert_eq!((x.clone() + y.clone()) * z.clone(), x * z.clone() + y * z);
    }

    #[test]
    fn poly_div(x in poly(0..100), y in poly(1..100)) {
        let mut r = x.clone();
        let q = r.div(&y);
        prop_assert_eq!(x, q * y + r);
    }
}

fn generator(t: usize) -> Poly<Gf<u8>> {
    let mut x = Poly(vec![Gf(1u8)]);
    for i in 1..=2 * t {
        x *= &Poly(vec![Gf::exp(i), Gf(1u8)]);
    }
    x
}

proptest! {
    #[test]
    fn test_generator_degree(t in 1..30usize) {
        let g = generator(t);
        prop_assert_eq!(g.degree(), 2 * t);
    }

    #[test]
    fn test_generator_root(t in 0..10usize) {
        let g = generator(t);
        for i in 1..=2 * t {
            let n = g.apply(Gf::exp(i));
            prop_assert_eq!(n.0, 0);
        }
        for i in 2 * t + 1..255 {
            let n = g.apply(Gf::exp(i));
            prop_assert_ne!(n.0, 0);
        }
    }
}

fn encode(m: &Poly<Gf<u8>>, t: usize) -> Poly<Gf<u8>> {
    // m(x) * x**e mod g(x) = C(x)
    let mut w = Poly(std::iter::repeat(Gf(0u8)).take(2 * t).collect());
    w.0.extend(&m.0);
    let mut r = w.clone();
    r %= &generator(t);
    w += &r;
    // w(x) = I(x) * x**e + C(x) = g(x) * q(x) for some q(x).
    w
}

fn syndrome(y: &Poly<Gf<u8>>, t: usize) -> Poly<Gf<u8>> {
    let mut s = vec![];
    for i in 1..=2 * t {
        s.push(y.apply(Gf::exp(i)));
    }
    yank_leading_zeros(&mut s);
    Poly(s)
}

prop_compose! {
    #[cfg(test)]
    fn msg_with_noise
      (msg_max_len: usize, t_max: usize)
      (msg in poly(1..msg_max_len), t in 1..=t_max)
      (noise in prop::collection::vec((0..msg.0.len() + 2 * t, any::<Gf<u8>>()), 1..=t), msg in Just(msg), t in Just(t))
      -> (Poly<Gf<u8>>, usize, Vec<(usize, Gf<u8>)>)
    {
        let noise = noise.into_iter().map(|(i, n)| {
            if n == Gf(0u8) {
                (i, Gf(1u8))
            } else {
                (i, n)
            }
        }).collect();
        (msg, t, noise)
    }
}

proptest! {
    #[test]
    fn test_syndrome_no_error(m in poly(1..30), t in 1..30usize) {
        let w = encode(&m, t);
        let s = syndrome(&w, t);
        prop_assert_eq!(s.0.len(), 0);
    }

    #[test]
    fn test_syndrome_error((m, t, noise) in msg_with_noise(30, 10)) {
        let mut w = encode(&m, t);
        for (i, n) in noise {
            w.0[i] += n;
        }
        let s = syndrome(&w, t);
        prop_assert_ne!(s.0.len(), 0);
    }
}

// solve key equation: given s(x) and t, outputs sigma(x) and eta(x) such that
//   sigma(x)*s(x) + phi(x)x**2t = eta(x)
//   sigma(0) = 1
fn solve(s: &Poly<Gf<u8>>, t: usize) -> (Poly<Gf<u8>>, Poly<Gf<u8>>) {
    let mut z = vec![Gf(0u8); 2 * t];
    z.push(Gf(1u8));
    // z = x**2t
    let z = Poly(z);
    // INVARIANT:
    // - r0 = a0 * x + b0 * y for some b0
    // - r1 = a1 * x + b1 * y for some b1
    let mut r0 = s.clone();
    let mut a0 = Poly(vec![Gf(1u8)]);
    let mut r1 = z;
    let mut a1 = Poly(vec![]);
    if r0.0.len() < r1.0.len() {
        std::mem::swap(&mut r0, &mut r1);
        std::mem::swap(&mut a0, &mut a1);
    }
    while r1.0.len() > t {
        let mut q = r0.div(&r1);
        std::mem::swap(&mut r0, &mut r1);
        q *= &a1;
        q += &a0;
        a0 = a1;
        a1 = q;
    }
    let k = a1.0[0].inv();
    a1.scale(k);
    r1.scale(k);
    (a1, r1)
}

proptest! {
    #[test]
    fn test_solve((m, t, noise) in msg_with_noise(30, 10)) {
        let mut w = encode(&m, t);
        for (i, n) in noise {
            w.0[i] += n;
        }
        let s = syndrome(&w, t);
        let (sigma, eta) = solve(&s, t);
        prop_assert!(sigma.degree() <= t);
        prop_assert!(eta.degree() < t);
        prop_assert_eq!(sigma.gcd(&eta), Poly(vec![Gf(1u8)]));
        prop_assert_eq!(sigma.0[0], Gf(1u8));
    }
}

fn correct(y: &mut Poly<Gf<u8>>, t: usize) {
    // Y(x) = w(x) + E(x) = g(x) * q(x) + E(x).
    // So, s[j] != 0 means E[j] != 0
    let s = syndrome(y, t);
    if !s.0.is_empty() {
        let (sigma, eta) = solve(&s, t);
        let mut sigmad = sigma.clone();
        sigmad.diff();
        for j in 0..y.0.len() {
            let ej = Gf::exp(j).inv();
            // Chien search
            if sigma.apply(ej).0 == 0 {
                // Forney algorithm
                y.0[j] += eta.apply(ej) / sigmad.apply(ej);
            }
        }
    }
}

proptest! {
    #[test]
    fn test_correct((m, t, noise) in msg_with_noise(30, 10)) {
        let w = encode(&m, t);
        let mut p = w.clone();
        for (i, n) in noise {
            p.0[i] += n;
        }
        correct(&mut p, t);
        prop_assert_eq!(w, p);
    }
}

fn decode(y: &Poly<Gf<u8>>, t: usize) -> Poly<Gf<u8>> {
    let mut y = y.clone();
    correct(&mut y, t);
    Poly(y.0[2 * t..].to_owned())
}

proptest! {
    #[test]
    fn encode_decode_rountrip((m, t, noise) in msg_with_noise(30, 10)) {
        let mut w = encode(&m, t);
        for (i, n) in noise {
            w.0[i] += n;
        }
        let n = decode(&w, t);
        prop_assert_eq!(m, n);
    }
}

fn encode_bytes(input: &[u8], t: usize) -> Vec<u8> {
    let m = Poly(input.iter().copied().map(Gf).collect());
    let m = encode(&m, t);
    m.0.into_iter().map(|n| n.0).collect()
}

fn decode_bytes(input: &[u8], t: usize) -> Vec<u8> {
    let m = Poly(input.iter().copied().map(Gf).collect());
    let m = decode(&m, t);
    m.0.into_iter().map(|n| n.0).collect()
}

fn main() {
    use std::io::{Read, Write};
    let mut args = std::env::args();
    args.next().unwrap();
    match args.next().unwrap().as_str() {
        "--encode" => {
            let t = args
                .next()
                .unwrap()
                .parse()
                .expect("--encode takes integer as an argument");
            let mut buf = vec![];
            std::io::stdin().read_to_end(&mut buf).expect("read_to_end");
            let enc = encode_bytes(&buf, t);
            std::io::stdout().write_all(&enc).expect("write_all");
        }
        "--decode" => {
            let t = args
                .next()
                .unwrap()
                .parse()
                .expect("--decode takes integer as an argument");
            let mut buf = vec![];
            std::io::stdin().read_to_end(&mut buf).expect("read_to_end");
            let enc = decode_bytes(&buf, t);
            std::io::stdout().write_all(&enc).expect("write_all");
        }
        arg => {
            panic!("unknown argument: {}", arg);
        }
    }
}
