#!/usr/bin/env python
"""
Brute-force interval motif arayici: her eserde tekrarlayan interval n-gram'lari
support ve null-model z-score ile skorlar. Once known_pattern_discovery, sonra bu.

Cikti:
  - results/stats/brute_force_pattern_summary.csv (eser bazinda ozet)
  - results/stats/brute_force_patterns_detail.csv (istege bagli, uzun)

Kullanim:
    python scripts/brute_force_patterns.py                    # motif analizi (paralel)
    python scripts/brute_force_patterns.py --chunk 0 --total-chunks 4   # chunk 0/4
    python scripts/brute_force_patterns.py --merge-chunks --total-chunks 4  # chunk'lari birlestir
    python scripts/brute_force_patterns.py --export-patterns  # 100 matematiksel oruntu -> CSV
    python scripts/brute_force_motif.py --chunk 0 --total-chunks 4  # direkt motif (ayri dosya)

Chunk: 4 terminalde chunk 0,1,2,3 ayri calistir, sonra --merge-chunks
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path


def main() -> None:
    """Motif analizi — brute_force_motif modülüne yönlendirir."""
    import sys
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir.parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location("brute_force_motif", scripts_dir / "brute_force_motif.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import argparse
    parser = argparse.ArgumentParser(description="Brute-force motif analizi")
    parser.add_argument("--chunk", type=int, default=None)
    parser.add_argument("--total-chunks", type=int, default=1)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--merge-chunks", action="store_true")
    args, _ = parser.parse_known_args()
    if args.merge_chunks:
        mod.merge_chunks(args.total_chunks)
    else:
        mod.run(chunk_index=args.chunk, total_chunks=args.total_chunks,
                no_parallel=args.no_parallel, max_workers=args.max_workers)


"""
100 MATEMATİKSEL ÖRÜNTÜ — MÜZİKAL HARMONİ İÇİN
=================================================
Her örüntü bir generator fonksiyonu döndürür.
Değerler frekans oranı, nota aralığı (semitone), ya da ritim ağırlığı olarak kullanılabilir.

KULLANIM ÖRNEĞI:
    seq = fibonacci_oranti(20)
    freqs = [220.0 * v for v in seq]  # A3 bazlı frekanslar
"""

import math
import itertools
from fractions import Fraction
from functools import reduce

# ─────────────────────────────────────────────
# KATEGORİ 1: KLASİK SAYISAL DİZİLER
# ─────────────────────────────────────────────

def p01_fibonacci(n):
    """Fibonacci dizisi: 1,1,2,3,5,8,13,...
    Armoni: Ardışık oran altın oran φ≈1.618'e yaklaşır → doğal gerilim-çözülüm"""
    a, b = 1, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

def p02_lucas(n):
    """Lucas dizisi: 2,1,3,4,7,11,18,...
    Armoni: Fibonacci'ye benzer ama farklı başlangıç → renkli varyasyon"""
    a, b = 2, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

def p03_tribonacci(n):
    """Tribonacci: 0,0,1,1,2,4,7,13,...
    Armoni: Üçlü öncüle bağımlılık → üçüncü sesle zenginleşen harmoni"""
    a, b, c = 0, 0, 1
    for _ in range(n):
        yield a
        a, b, c = b, c, a + b + c

def p04_padovan(n):
    """Padovan dizisi: 1,1,1,2,2,3,4,5,7,...  (P(n)=P(n-2)+P(n-3))
    Armoni: Spiral büyüme → daha yavaş açılan gerilimler"""
    seq = [1, 1, 1]
    for i in range(3, n):
        seq.append(seq[-2] + seq[-3])
    yield from seq[:n]

def p05_pell(n):
    """Pell dizisi: 0,1,2,5,12,29,70,...
    Armoni: √2'ye yaklaşan oranlar → tritone (azaltılmış 5li) eğilimi"""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, 2 * b + a

def p06_jacobsthal(n):
    """Jacobsthal: 0,1,1,3,5,11,21,43,...  (J(n)=J(n-1)+2*J(n-2))
    Armoni: Katlanma ve geri dönüş → yankılanan ritim"""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, b + 2 * a

def p07_perrin(n):
    """Perrin dizisi: 3,0,2,3,2,5,5,7,...  (P(n)=P(n-2)+P(n-3))
    Armoni: Kaotik başlangıç → stabil renge kavuşma"""
    seq = [3, 0, 2]
    for i in range(3, n):
        seq.append(seq[-2] + seq[-3])
    yield from seq[:n]

def p08_stern_brocot(n):
    """Stern-Brocot dizisi: 1,1,2,1,3,2,3,1,4,...
    Armoni: Tüm rasyonel oranları kapsar → kromatik armoni"""
    seq = [1]
    while len(seq) < n:
        new_seq = []
        for i in range(len(seq) - 1):
            new_seq.append(seq[i])
            new_seq.append(seq[i] + seq[i + 1])
        new_seq.append(seq[-1])
        seq = new_seq
    yield from seq[:n]

def p09_sylvester(n):
    """Sylvester dizisi: 2,3,7,43,1807,...  (s(n)=s(n-1)*(s(n-1)-1)+1)
    Armoni: Süper üstel büyüme → giderek şiddetlenen ses patlamaları"""
    s = 2
    for _ in range(n):
        yield s
        s = s * (s - 1) + 1

def p10_catalan(n):
    """Catalan sayıları: 1,1,2,5,14,42,132,...
    Armoni: Parantez yapısı → iç içe geçmiş melodik yapılar"""
    c = 1
    for i in range(n):
        yield c
        c = c * 2 * (2 * i + 1) // (i + 2)


# ─────────────────────────────────────────────
# KATEGORİ 2: ASAL SAYILAR VE MODÜLEr ARİTMETİK
# ─────────────────────────────────────────────

def p11_asal_sayilar(n):
    """Asal sayılar: 2,3,5,7,11,13,...
    Armoni: Bölünsüzlük → disonans frekanslar"""
    def is_prime(num):
        if num < 2: return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0: return False
        return True
    count, num = 0, 2
    while count < n:
        if is_prime(num):
            yield num
            count += 1
        num += 1

def p12_asal_bosluklar(n):
    """Asal sayı boşlukları: 1,2,2,4,2,4,2,4,6,...
    Armoni: Ritimsel sürpriz → beklenmedik vuruşlar"""
    primes = list(p11_asal_sayilar(n + 1))
    yield primes[0]
    for i in range(1, n):
        yield primes[i] - primes[i - 1]

def p13_mod_fibonacci(n, mod=12):
    """Fibonacci mod 12 (kromatik skala): 1,1,2,3,5,8,1,9,10,7,5,0,...
    Armoni: 12-TET sistemi üzerinde döngüsel Fibonacci"""
    a, b = 1, 1
    for _ in range(n):
        yield a % mod
        a, b = b, (a + b) % mod

def p14_mod_geometrik(n, r=3, mod=7):
    """Geometrik dizi mod 7 (diyatonik): 1,3,2,6,4,5,1,...
    Armoni: Majör skaladaki derece döngüsü"""
    val = 1
    for _ in range(n):
        yield val
        val = (val * r) % mod

def p15_collatz(n, start=27):
    """Collatz dizisi (3n+1): 27,82,41,124,...,1
    Armoni: Kaotik ama sonlu → sürprizlerle dolu melodi"""
    val = start
    count = 0
    while count < n:
        yield val
        val = val // 2 if val % 2 == 0 else 3 * val + 1
        count += 1

def p16_totient(n):
    """Euler Totient φ(n): 1,1,2,2,4,2,6,4,6,4,...
    Armoni: Evrensel mod çemberi → polifoniye uygun ritim"""
    def phi(num):
        result = num
        p = 2
        temp = num
        while p * p <= temp:
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
            p += 1
        if temp > 1:
            result -= result // temp
        return result
    for i in range(1, n + 1):
        yield phi(i)

def p17_lucky_sayilar(n):
    """Şanslı sayılar: 1,3,7,9,13,15,21,25,...
    Armoni: Elek örüntüsü → seçilmiş notalar"""
    sieve = list(range(1, 4 * n))
    i = 1
    while i < len(sieve) and i < n:
        step = sieve[i]
        sieve = [x for j, x in enumerate(sieve) if (j + 1) % step != 0]
        i += 1
    yield from sieve[:n]

def p18_golomb(n):
    """Golomb dizisi: 1,2,2,3,3,4,4,4,5,5,5,...
    Armoni: Kendi kendini tanımlayan sıra → fraktal melodik tekrar"""
    g = [0, 1]
    for i in range(2, n + 1):
        g.append(1 + g[i - g[g[i - 1]]])
    yield from g[1:n + 1]

def p19_recaman(n):
    """Recamán dizisi: 0,1,3,6,2,7,13,20,...
    Armoni: Geri gitmek veya ileri gitmek → melodik karar anları"""
    seen = set()
    val = 0
    for i in range(n):
        yield val
        seen.add(val)
        prev = val
        candidate = val - i
        if candidate > 0 and candidate not in seen:
            val = candidate
        else:
            val = prev + i

def p20_kaprekar_6174(n):
    """Kaprekar dönüşümü mod dizisi
    Armoni: Her 4 basamaklı sayı 6174'e yaklaşır → kaçınılmaz çözüm"""
    val = 1234
    for _ in range(n):
        yield val
        digits = sorted(f"{val:04d}")
        big = int("".join(reversed(digits)))
        small = int("".join(digits))
        val = big - small
        if val == 0: val = 1000


# ─────────────────────────────────────────────
# KATEGORİ 3: GEOMETRİK VE TRİGONOMETRİK ÖRÜNTÜLER
# ─────────────────────────────────────────────

def p21_harmonik_seri(n):
    """Harmonik seri: 1, 1/2, 1/3, 1/4, ...
    Armoni: Doğal üst sesler → akustik overtone serisi"""
    for i in range(1, n + 1):
        yield Fraction(1, i)

def p22_overtone_serisi(n, temel=220.0):
    """Overtone (üst ses) frekansları: f, 2f, 3f, 4f, 5f,...
    Armoni: Fiziğin kendisi — tüm harmoninin temeli"""
    for i in range(1, n + 1):
        yield temel * i

def p23_just_intonation(n):
    """Just intonation oranları (5-limit): 1/1, 9/8, 5/4, 4/3, 3/2,...
    Armoni: Saf aralıklar → tam rezonans"""
    oranlar = [
        Fraction(1, 1), Fraction(9, 8), Fraction(5, 4),
        Fraction(4, 3), Fraction(3, 2), Fraction(5, 3),
        Fraction(15, 8), Fraction(2, 1)
    ]
    for i in range(n):
        yield oranlar[i % len(oranlar)]

def p24_pythagorean_scale(n):
    """Pisagor skala: (3/2)^k mod oktav
    Armoni: Beşli çevre → klasik Batı armoninin temeli"""
    oran = Fraction(3, 2)
    val = Fraction(1, 1)
    for _ in range(n):
        while val >= 2:
            val /= 2
        while val < 1:
            val *= 2
        yield val
        val *= oran

def p25_sin_harmonikleri(n, harmonik=5, rate=100):
    """Sinüs dalgası örnekleri: sin(2πkt/N)
    Armoni: Saf ton üretimi"""
    for i in range(n):
        yield math.sin(2 * math.pi * harmonik * i / rate)

def p26_lissajous_x(n, a=3, b=2):
    """Lissajous x koordinatları: sin(at)
    Armoni: a:b oranı müzikal aralığa karşılık gelir"""
    for i in range(n):
        t = 2 * math.pi * i / n
        yield math.sin(a * t)

def p27_lissajous_y(n, a=3, b=2):
    """Lissajous y koordinatları: sin(bt+δ)
    Armoni: Beşli (3:2) → stabil oval, majör üçlü (5:4) → karmaşık"""
    for i in range(n):
        t = 2 * math.pi * i / n
        yield math.sin(b * t + math.pi / 4)

def p28_chebyshev_harmonik(n, derece=5):
    """Chebyshev polinomu değerleri: T_n(cos θ) = cos(nθ)
    Armoni: Köprü harmonikleri → eşit tampere benzer örüntü"""
    for i in range(n):
        x = math.cos(math.pi * i / n)
        yield math.cos(derece * math.acos(max(-1, min(1, x))))

def p29_fourier_kare_dalga(n, harmonikler=5):
    """Kare dalga Fourier açılımı: Σ sin((2k-1)t)/(2k-1)
    Armoni: Tek harmonikler → tiz, zengin tını"""
    for i in range(n):
        t = 2 * math.pi * i / n
        val = sum(math.sin((2 * k - 1) * t) / (2 * k - 1)
                  for k in range(1, harmonikler + 1))
        yield val

def p30_zararlintili_titresim(n, f1=5, f2=5.5, rate=100):
    """Titreşim (beat): cos(2π f1 t) + cos(2π f2 t)
    Armoni: Frekans farkı = titreşim hızı → algılanan ritim"""
    for i in range(n):
        t = i / rate
        yield math.cos(2 * math.pi * f1 * t) + math.cos(2 * math.pi * f2 * t)


# ─────────────────────────────────────────────
# KATEGORİ 4: FRAKTAL VE KAOTİK SİSTEMLER
# ─────────────────────────────────────────────

def p31_logistik_harita(n, r=3.7, x0=0.5):
    """Lojistik harita: x_{n+1} = r*x_n*(1-x_n)
    Armoni: r>3.57 → kaos; melodik tahmin edilemezlik"""
    x = x0
    for _ in range(n):
        yield x
        x = r * x * (1 - x)

def p32_tent_harita(n, mu=1.9, x0=0.3):
    """Tent harita: doğrusal parçalı kaos
    Armoni: Sert geçişler → percussive vuruşlar"""
    x = x0
    for _ in range(n):
        yield x
        x = mu * min(x, 1 - x)

def p33_lorenz_x(n, sigma=10, rho=28, beta=8/3, dt=0.01):
    """Lorenz çekicisi x bileşeni
    Armoni: Kaotik ama yapılı → avant-garde melodi"""
    x, y, z = 1.0, 1.0, 1.0
    for _ in range(n):
        yield x
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt

def p34_henon_x(n, a=1.4, b=0.3):
    """Hénon haritası x bileşeni
    Armoni: Garip çekici → tekrarlanmayan ama benzer motifler"""
    x, y = 0.0, 0.0
    for _ in range(n):
        yield x
        x, y = 1 - a * x**2 + y, b * x

def p35_arnolds_cat(n, mod=12):
    """Arnold'un kedi haritası mod 12
    Armoni: Ergodik → tüm 12 tonu ziyaret eder"""
    x, y = 1, 0
    for _ in range(n):
        yield x
        x, y = (x + y) % mod, (x + 2 * y) % mod

def p36_sierpinski_toplami(n):
    """Sierpinski toplamı (binomial katsayılar mod 2)
    Armoni: Fraktal ritim → öz-benzer vuruş kalıpları"""
    for i in range(n):
        row_sum = 0
        for k in range(i + 1):
            c = 1
            for j in range(k):
                c = c * (i - j) // (j + 1)
            row_sum += c % 2
        yield row_sum

def p37_dragon_curve(n):
    """Dragon eğrisi katlama dizisi: L,L,R,L,L,R,R,...
    Armoni: ±1 olarak → melodik yön değişimleri"""
    seq = [1]
    for _ in range(n - 1):
        mid = [1]
        folded = [(-1 if i % 2 == 0 else 1) * seq[i // 2]
                  for i in range(len(seq) * 2)]
        seq = seq + [1] + list(reversed(seq))
        if len(seq) >= n:
            break
    yield from (1 if x > 0 else -1 for x in seq[:n])

def p38_thue_morse(n):
    """Thue-Morse dizisi: 0,1,1,0,1,0,0,1,...
    Armoni: Kübik olmayan dizi → anti-ritimik vuruş"""
    for i in range(n):
        yield bin(i).count('1') % 2

def p39_cantor_seti(n, derinlik=6):
    """Cantor seti varlık dizisi (0 veya 1)
    Armoni: Boşluklar ve sesler → minimalist müzik"""
    def cantor(start, end, depth):
        if depth == 0:
            return [(start, end)]
        third = (end - start) / 3
        left = cantor(start, start + third, depth - 1)
        right = cantor(end - third, end, depth - 1)
        return left + right
    segments = cantor(0, 1, derinlik)
    result = []
    for i in range(n):
        t = i / n
        in_cantor = any(s <= t <= e for s, e in segments)
        result.append(1 if in_cantor else 0)
    yield from result

def p40_mandelbrot_iterasyon(n, c_real=-0.7, c_imag=0.27):
    """Mandelbrot iterasyon sayıları
    Armoni: Sınır bölgesi → zengin harmonik renk"""
    c = complex(c_real, c_imag)
    z = 0
    for _ in range(n):
        yield abs(z)
        if abs(z) > 2:
            z = 0
        z = z * z + c


# ─────────────────────────────────────────────
# KATEGORİ 5: MÜZİKSEL TEORİ TEMELLİ
# ─────────────────────────────────────────────

def p41_bes_cember(n):
    """Beşli çember: C=0, G=7, D=2, A=9, ... semitone cinsinden
    Armoni: Tonalite haritası → modülasyon yolları"""
    cember = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
    for i in range(n):
        yield cember[i % 12]

def p42_altin_oran_melodisi(n, temel=220.0):
    """Altın oran üreteci: f_{n+1} = f_n * φ mod oktav
    Armoni: Φ≈1.618 → doğal oran, hiçbir frekans tekrarlanmaz"""
    phi = (1 + math.sqrt(5)) / 2
    freq = temel
    for _ in range(n):
        yield freq
        freq *= phi
        while freq > temel * 2:
            freq /= 2

def p43_12tet_araliklari(n):
    """12-TET aralık oranları: 2^(k/12)
    Armoni: Eşit tampere sistemi → tüm tonlar aynı mesafe"""
    for i in range(n):
        yield 2 ** ((i % 12) / 12)

def p44_modal_skala(n, mod="dorian"):
    """Modal skala aralıkları (semitone)
    Armoni: Farklı karakterler → müzikal renk paleti"""
    modlar = {
        "ionian":     [0, 2, 4, 5, 7, 9, 11],
        "dorian":     [0, 2, 3, 5, 7, 9, 10],
        "phrygian":   [0, 1, 3, 5, 7, 8, 10],
        "lydian":     [0, 2, 4, 6, 7, 9, 11],
        "mixolydian": [0, 2, 4, 5, 7, 9, 10],
        "aeolian":    [0, 2, 3, 5, 7, 8, 10],
        "locrian":    [0, 1, 3, 5, 6, 8, 10],
    }
    scale = modlar.get(mod, modlar["dorian"])
    for i in range(n):
        oktav = i // len(scale)
        yield scale[i % len(scale)] + 12 * oktav

def p45_pentatonik(n):
    """Pentatonik skala aralıkları
    Armoni: 5 nota → evrensel uyum, disonans yok"""
    scale = [0, 2, 4, 7, 9]
    for i in range(n):
        oktav = i // 5
        yield scale[i % 5] + 12 * oktav

def p46_mikrotonal_41tet(n):
    """41-TET mikrotonal sistem
    Armoni: Ultra ince aralıklar → Orta Doğu ve Hint müziği"""
    for i in range(n):
        yield 2 ** ((i % 41) / 41)

def p47_rytim_euclidean(n, pulses=5, steps=8):
    """Öklid ritmi: E(5,8) = [1,0,1,1,0,1,1,0]
    Armoni: Tresillo, clave ritmi → Afrika ve Latin pols"""
    pattern = []
    bucket = 0
    for i in range(steps):
        bucket += pulses
        if bucket >= steps:
            bucket -= steps
            pattern.append(1)
        else:
            pattern.append(0)
    for i in range(n):
        yield pattern[i % steps]

def p48_zarb_ritim(n):
    """Zarb (4/4 içinde 3+3+2) poliritmik bölünme
    Armoni: Hemiola → gerilim yaratır"""
    pattern = [3, 3, 2]
    pos = 0
    for _ in range(n):
        yield pattern[pos % 3]
        pos += 1

def p49_phi_ritim(n):
    """Altın oran ritmi: Beatler φ biriminde
    Armoni: Oransal ritim → tam natural hissiyat"""
    phi = (1 + math.sqrt(5)) / 2
    beats = []
    t = 0
    while len(beats) < n:
        beats.append(round(t, 4))
        t += phi
    yield from beats

def p50_boulez_seri(n):
    """Boulez tarz pitch serisi permütasyonlar
    Armoni: 12-ton serializm → sistematik disonans"""
    base = [0, 11, 3, 4, 8, 7, 9, 5, 6, 1, 10, 2]
    for i in range(n):
        yield base[i % 12]


# ─────────────────────────────────────────────
# KATEGORİ 6: KOMBİNATORİK VE DİZİ TEORİSİ
# ─────────────────────────────────────────────

def p51_pascal_ucgeni_satir(n, satir=10):
    """Pascal üçgeni satır değerleri
    Armoni: Binomial katsayılar → kümülatifit ses yoğunluğu"""
    c = 1
    for k in range(n):
        yield c
        if k < satir:
            c = c * (satir - k) // (k + 1)
        else:
            c = 0

def p52_motzkin(n):
    """Motzkin sayıları: 1,1,2,4,9,21,51,...
    Armoni: Köprü sayıları → melodik arc yapısı"""
    M = [1, 1]
    for i in range(2, n):
        M.append(((2 * i + 2) * M[i - 1] + (3 * i - 3) * M[i - 2]) // (i + 3))
    yield from M[:n]

def p53_narayana(n, k=2):
    """Narayana N(n,k): üçgen sayı alanı
    Armoni: k-sesli armoni olasılıkları"""
    def nara(n, k):
        def C(n, r):
            if r > n or r < 0: return 0
            r = min(r, n - r)
            result = 1
            for i in range(r):
                result = result * (n - i) // (i + 1)
            return result
        return C(n, k) * C(n, k - 1) // n if n > 0 else 1
    for i in range(1, n + 1):
        yield nara(i, min(k, i))

def p54_stirling_1(n):
    """Stirling 1. tür sayıları (işaretli)
    Armoni: Permütasyon döngüleri → melodik döngü sayıları"""
    def s1(n, k):
        if n == 0 and k == 0: return 1
        if n == 0 or k == 0: return 0
        return (n - 1) * s1(n - 1, k) + s1(n - 1, k - 1)
    for i in range(1, n + 1):
        yield s1(i, max(1, i // 2))

def p55_bell_sayilari(n):
    """Bell sayıları: 1,1,2,5,15,52,...
    Armoni: Bölümleme sayıları → akor rengi kombinasyonları"""
    bell = [[0] * (n + 1) for _ in range(n + 1)]
    bell[0][0] = 1
    for i in range(1, n + 1):
        bell[i][0] = bell[i - 1][i - 1]
        for j in range(1, i + 1):
            bell[i][j] = bell[i - 1][j - 1] + bell[i][j - 1]
    for i in range(n):
        yield bell[i][0]

def p56_gray_kodu(n):
    """Gray kodu: 0,1,3,2,6,7,5,4,...
    Armoni: Bitişik değerler 1 bit farklı → pürüzsüz geçişler"""
    for i in range(n):
        yield i ^ (i >> 1)

def p57_rudin_shapiro(n):
    """Rudin-Shapiro dizisi: ±1
    Armoni: Düzgün spektrum dağılımı → beyaz gürültü benzeri"""
    def rs(n):
        count = 0
        x = n
        while x > 0:
            if x & 3 == 3:
                count += 1
            x >>= 1
        return 1 if count % 2 == 0 else -1
    for i in range(n):
        yield rs(i)

def p58_baum_sweet(n):
    """Baum-Sweet dizisi: 0 veya 1
    Armoni: Seyrek ses → minimal müzik"""
    def bs(n):
        if n == 0: return 1
        while n % 2 == 0:
            n //= 2
        s = bin(n)[2:]
        i = 0
        while i < len(s):
            if s[i] == '0':
                run = 0
                while i < len(s) and s[i] == '0':
                    run += 1
                    i += 1
                if run % 2 == 1:
                    return 0
            else:
                i += 1
        return 1
    for i in range(n):
        yield bs(i)

def p59_kolakoski(n):
    """Kolakoski dizisi: 1,2,2,1,1,2,1,2,2,...
    Armoni: Kendi karakter uzunluklarını üretir → öz-benzer ritim"""
    result = [1, 2]
    i = 1
    while len(result) < n:
        val = result[i]
        result.extend([result[-1] ^ 3] * (val - 1))
        result.append(result[-1] ^ 3)
        i += 1
    yield from result[:n]

def p60_lempel_ziv(n, seed=0b1101001011):
    """Lempel-Ziv karmaşıklık dizisi (bit dizisi)
    Armoni: Sıkıştırma karmaşıklığı → bilgi yoğunluğu"""
    bits = list(map(int, bin(seed)[2:]))
    while len(bits) < n:
        bits.append((bits[-1] + bits[-2]) % 2)
    yield from bits[:n]


# ─────────────────────────────────────────────
# KATEGORİ 7: FİZİKSEL VE DALGA SİSTEMLERİ
# ─────────────────────────────────────────────

def p61_kuantum_harmonik(n, omega=1.0, hbar=1.0):
    """Kuantum harmonik osilatör enerji seviyeleri: E_n = ℏω(n+1/2)
    Armoni: Ayrık enerji → ayrık frekans basamakları"""
    for i in range(n):
        yield hbar * omega * (i + 0.5)

def p62_dalga_paketi(n, k0=5, sigma=0.5, rate=100):
    """Gaussian dalga paketi: e^(-x²/2σ²) * cos(k0*x)
    Armoni: Lokalleşmiş ton → belirli frekans etrafında titreşim"""
    for i in range(n):
        x = (i - n / 2) / rate
        envelope = math.exp(-(x**2) / (2 * sigma**2))
        yield envelope * math.cos(k0 * x)

def p63_tam_dalga_rektifire(n, f=3):
    """Tam dalga doğrultucu: |sin(ft)|
    Armoni: Katlanmış sinüs → tüm harmonikler çift"""
    for i in range(n):
        t = 2 * math.pi * i / n
        yield abs(math.sin(f * t))

def p64_fm_sentezi(n, fc=5, fm=3, mi=2):
    """FM sentezi: sin(2π fc t + mi * sin(2π fm t))
    Armoni: Modülasyon indeksi → tını zenginliği"""
    for i in range(n):
        t = i / n
        yield math.sin(2 * math.pi * fc * t + mi * math.sin(2 * math.pi * fm * t))

def p65_am_sentezi(n, fc=10, fm=1):
    """AM sentezi: (1 + cos(2π fm t)) * cos(2π fc t)
    Armoni: Yan bantlar → tını rengi"""
    for i in range(n):
        t = i / n
        envelope = 1 + math.cos(2 * math.pi * fm * t)
        yield envelope * math.cos(2 * math.pi * fc * t)

def p66_duffing_osilatoru(n, alpha=1, beta=-1, delta=0.3, gamma=0.5, dt=0.05):
    """Duffing osilatörü: çift-kuyu kaotik sistem
    Armoni: İki stabil durum arası geçiş → dramatik ton atlamaları"""
    x, v = 0.1, 0.0
    omega = 1.2
    for i in range(n):
        yield x
        ax = x - beta * x**3 - delta * v + gamma * math.cos(omega * i * dt)
        v += ax * dt
        x += v * dt

def p67_van_der_pol(n, mu=2.0, dt=0.05):
    """Van der Pol osilatörü: limit cycle
    Armoni: Stabil döngü → sürdürülebilir titreşim"""
    x, y = 0.5, 0.5
    for _ in range(n):
        yield x
        dx = y
        dy = mu * (1 - x**2) * y - x
        x += dx * dt
        y += dy * dt

def p68_kuramoto_model(n, omega_list=None, K=2.0, dt=0.1):
    """Kuramoto modeli fazları: senkronizasyon
    Armoni: Bağlantılı osilatörler → doğal senkronizasyon"""
    if omega_list is None:
        omega_list = [1.0, 1.2, 0.8, 1.5, 0.9]
    N = len(omega_list)
    phases = [math.pi * i / N for i in range(N)]
    for _ in range(n):
        yield sum(phases) / N  # Ortalama faz
        new_phases = []
        for i in range(N):
            coupling = K / N * sum(math.sin(phases[j] - phases[i]) for j in range(N))
            new_phases.append(phases[i] + (omega_list[i] + coupling) * dt)
        phases = new_phases

def p69_geri_beslemeli_gecikme(n, tau=3, a=0.25, b=0.1):
    """Gecikmeli geri besleme: x(t) = a*x(t-τ) - b*x(t-2τ)
    Armoni: Yankı ve gecikme → doğal reverb"""
    buf = [0.5] * (2 * tau)
    for i in range(n):
        val = a * buf[-tau] - b * buf[-2 * tau] + 0.01 * math.sin(i * 0.3)
        yield val
        buf.append(val)

def p70_wavelets_haar(n):
    """Haar wavelet katsayıları
    Armoni: Ani geçişler → percussive element"""
    level = n.bit_length()
    for i in range(n):
        # Basit Haar: blok içinde +1/-1
        block_size = 2 ** (level - 1 - i.bit_length() + 1) if i > 0 else n
        yield 1 if (i // max(1, block_size // 2)) % 2 == 0 else -1


# ─────────────────────────────────────────────
# KATEGORİ 8: SAYILAR TEORİSİ VE CEBİRSEL
# ─────────────────────────────────────────────

def p71_mersenne_esik(n):
    """Mersenne asal benzeri: 2^p - 1
    Armoni: Büyük asal → ezici disonans ani çözüm"""
    primes_p = [2, 3, 5, 7, 13, 17, 19, 31]
    for i in range(min(n, len(primes_p))):
        yield 2 ** primes_p[i] - 1

def p72_tam_sayilar(n):
    """Mükemmel sayılar (ve yakın): 6,28,496,8128,...
    Armoni: σ(n)=2n → dengeli armoni metaforu"""
    def is_perfect(num):
        return sum(i for i in range(1, num) if num % i == 0) == num
    count, num = 0, 2
    while count < n:
        if is_perfect(num):
            yield num
            count += 1
        num += 1
        if count == 0 and num > 10000:
            yield from [6, 28, 496, 8128, 33550336][:n]
            break

def p73_liouville_lambda(n):
    """Liouville fonksiyonu λ(n): ±1
    Armoni: Asal faktör sayısı paritesi → dinamik ters çevirme"""
    def liouville(num):
        count = 0
        d = 2
        while d * d <= num:
            while num % d == 0:
                count += 1
                num //= d
            d += 1
        if num > 1:
            count += 1
        return 1 if count % 2 == 0 else -1
    for i in range(1, n + 1):
        yield liouville(i)

def p74_mobius_mu(n):
    """Möbius fonksiyonu μ(n): -1, 0, 1
    Armoni: Üç seviyeli dinamik — sessizlik, piano, forte"""
    def mobius(num):
        if num == 1: return 1
        factors = []
        d = 2
        while d * d <= num:
            if num % d == 0:
                count = 0
                while num % d == 0:
                    count += 1
                    num //= d
                if count > 1: return 0
                factors.append(d)
            d += 1
        if num > 1: factors.append(num)
        return (-1) ** len(factors)
    for i in range(1, n + 1):
        yield mobius(i)

def p75_sigma_bolucusu(n):
    """Sigma fonksiyonu σ(n): bölücüler toplamı
    Armoni: Sayısal zenginlik → harmonik zenginlik"""
    for i in range(1, n + 1):
        yield sum(d for d in range(1, i + 1) if i % d == 0)

def p76_pisagor_uclu(n):
    """Pisagor üçlüleri hipotenüsü: 5,10,13,15,17,20,...
    Armoni: Dik üçgen → mod 12 aralık hesabı"""
    hyps = set()
    for a in range(1, 10 * n):
        for b in range(a, 10 * n):
            c2 = a * a + b * b
            c = int(c2 ** 0.5)
            if c * c == c2:
                hyps.add(c)
    sorted_h = sorted(hyps)[:n]
    yield from sorted_h

def p77_babelon_oruntuleri(n):
    """Babilon kil tableti sayı sistemi (base-60 mod)
    Armoni: 60 = LCM(1..5) → beşli harmonik doluluk"""
    for i in range(n):
        val = i % 60
        digit1 = val // 10
        digit2 = val % 10
        yield digit1 * 10 + digit2

def p78_egyptian_fraction(n, target=None):
    """Mısır kesirleri: 1 = 1/2 + 1/3 + 1/6 ...
    Armoni: Harmonik bölünme → üst ses ayrıştırma"""
    if target is None:
        target = Fraction(1, 1)
    remaining = target
    count = 0
    d = 1
    while count < n and remaining > 0:
        frac = Fraction(1, d)
        if frac <= remaining:
            yield frac
            remaining -= frac
            count += 1
        d += 1

def p79_zeckendorf(n):
    """Zeckendorf gösterimi (Fibonacci bazı)
    Armoni: Her sayı benzersiz Fibonacci toplamı → tekrar yok"""
    fibs = list(p01_fibonacci(20))
    for num in range(n):
        rep = []
        remaining = num
        for f in reversed(fibs):
            if f <= remaining:
                rep.append(f)
                remaining -= f
        yield sum(rep)

def p80_farey_dizisi(n, order=8):
    """Farey dizisi F_n: sıralı rasyoneller [0,1] içinde
    Armoni: Komşu oranlar → pürüzsüz ses hareketi"""
    farey = []
    a, b, c, d = 0, 1, 1, order
    farey.append(Fraction(a, b))
    while c <= order:
        k = (order + b) // d
        a, b, c, d = c, d, k * c - a, k * d - b
        farey.append(Fraction(a, b))
    yield from farey[:n]


# ─────────────────────────────────────────────
# KATEGORİ 9: AUTOMATA VE HÜCRESEL SİSTEMLER
# ─────────────────────────────────────────────

def p81_kural_30(n, width=20):
    """Wolfram Kural 30 (kaotik hücresel otomat)
    Armoni: Deterministik kaos → pseudo-random melodi"""
    state = [0] * width
    state[width // 2] = 1
    rule = 30
    for _ in range(n):
        yield sum(state)  # toplam aktif hücre = şiddet
        new_state = []
        for i in range(width):
            left = state[(i - 1) % width]
            center = state[i]
            right = state[(i + 1) % width]
            pattern = (left << 2) | (center << 1) | right
            new_state.append((rule >> pattern) & 1)
        state = new_state

def p82_kural_90(n, width=20):
    """Wolfram Kural 90 (Sierpinski üçgeni)
    Armoni: XOR tabanlı fraktal → geometrik ritim"""
    state = [0] * width
    state[width // 2] = 1
    for _ in range(n):
        yield sum(state)
        new_state = [
            state[(i - 1) % width] ^ state[(i + 1) % width]
            for i in range(width)
        ]
        state = new_state

def p83_game_of_life_canli(n, grid_size=8):
    """Game of Life canlı hücre sayısı
    Armoni: Nüfus dalgalanması → organik ritim"""
    grid = [[1 if (i + j) % 3 == 0 else 0
             for j in range(grid_size)] for i in range(grid_size)]
    for _ in range(n):
        yield sum(sum(row) for row in grid)
        new_grid = []
        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                neighbors = sum(
                    grid[(i + di) % grid_size][(j + dj) % grid_size]
                    for di in [-1, 0, 1] for dj in [-1, 0, 1]
                    if (di, dj) != (0, 0)
                )
                alive = grid[i][j]
                row.append(1 if (alive and neighbors in [2, 3])
                           or (not alive and neighbors == 3) else 0)
            new_grid.append(row)
        grid = new_grid

def p84_sandpile_toplami(n, start_chips=100):
    """Sandpile modeli (Abelian): yığılma ve çığlanma
    Armoni: Kritik eşikte çığlanma → dramatik crescendo"""
    grid_size = 5
    grid = [[0] * grid_size for _ in range(grid_size)]
    cx, cy = grid_size // 2, grid_size // 2
    grid[cx][cy] = start_chips
    for _ in range(n):
        yield sum(sum(row) for row in grid)
        tipped = False
        for i in range(grid_size):
            for j in range(grid_size):
                if grid[i][j] >= 4:
                    grid[i][j] -= 4
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid_size and 0 <= nj < grid_size:
                            grid[ni][nj] += 1
                    tipped = True
        if not tipped:
            grid[cx][cy] += 10

def p85_xorshift(n, seed=12345):
    """XOR-shift PRNG dizisi mod 1024
    Armoni: Hızlı pseudo-random → algoritmik kompozisyon"""
    x = seed
    for _ in range(n):
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
        x &= 0xFFFFFFFF
        yield x % 1024


# ─────────────────────────────────────────────
# KATEGORİ 10: İLERİ ÖRÜNTÜLER
# ─────────────────────────────────────────────

def p86_hilbert_egri_x(n):
    """Hilbert eğrisi x koordinatları
    Armoni: Alan dolduran eğri → spektral tarama"""
    def d2xy(n, d):
        x = y = 0
        s = 1
        while s < n:
            rx = 1 if (d & 2) else 0
            ry = 1 if (d & 1) ^ rx else 0
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            d >>= 2
            s <<= 1
        return x, y
    size = 2 ** math.ceil(math.log2(n))
    for i in range(n):
        x, _ = d2xy(size, i)
        yield x

def p87_vicsek_fraktal(n, adim=4):
    """Vicsek fraktal boyutu yaklaşımı
    Armoni: Öz-benzer büyüme → melodik öz-referans"""
    def vicsek_count(level):
        if level == 0: return 1
        return 5 * vicsek_count(level - 1)
    for i in range(n):
        yield vicsek_count(i % adim)

def p88_voronoi_mesafe(n, nokta_sayisi=5):
    """Voronoi bölge mesafeleri (1D)
    Armoni: En yakın merkez → ton atama algoritması"""
    import random
    random.seed(42)
    centers = sorted([random.random() for _ in range(nokta_sayisi)])
    for i in range(n):
        x = i / n
        dist = min(abs(x - c) for c in centers)
        yield dist

def p89_lindenmayer_uzunluk(n):
    """L-sistemi (bitki büyümesi) sembol sayısı
    Armoni: F→FF+[+F-F-F]-[-F+F+F] → fraktal büyüme ritmi"""
    s = "F"
    kurallar = {"F": "FF+[+F-F-F]-[-F+F+F]"}
    for i in range(n):
        yield len(s)
        if i < n - 1:
            s = "".join(kurallar.get(c, c) for c in s)
            if len(s) > 10**6:  # Hafıza koruyucu
                s = s[:1000]

def p90_quasicrystal_1d(n, tau=None):
    """1D Kvaziperiyodik dizi (Fibonacci sözcüğü)
    Armoni: Asla tekrar etmez ama düzenli → pentatonik benzeri özgürlük"""
    if tau is None:
        tau = (1 + math.sqrt(5)) / 2
    for i in range(n):
        yield 0 if math.floor((i + 1) * tau) - math.floor(i * tau) == 1 else 1

def p91_prime_spiral_aci(n):
    """Asal sayı spiralin açısı (Ulam spirali)
    Armoni: Asal dağılım → doğrusal olmayan melodik yol"""
    def is_prime(num):
        if num < 2: return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0: return False
        return True
    for i in range(1, n + 1):
        if is_prime(i):
            yield (i * math.sqrt(i)) % (2 * math.pi)
        else:
            yield 0.0

def p92_hofstadter_Q(n):
    """Hofstadter Q dizisi: Q(n) = Q(n-Q(n-1)) + Q(n-Q(n-2))
    Armoni: Kendi geçmişine başvuru → müzikal öz-alıntı"""
    Q = [0, 1, 1]
    for i in range(3, n + 1):
        Q.append(Q[i - Q[i - 1]] + Q[i - Q[i - 2]])
    yield from Q[1:n + 1]

def p93_hofstadter_G(n):
    """Hofstadter G dizisi: G(n) = n - G(G(n-1))
    Armoni: Metalik oran yaklaşımı → altın oran melodik"""
    G = [0]
    for i in range(1, n + 1):
        G.append(i - G[G[i - 1]])
    yield from G[1:n + 1]

def p94_sicherman_zar(n):
    """Sicherman zar olasılık dağılımı
    Armoni: Aynı dağılım farklı sayılar → yeniden etiketleme"""
    zar1 = [1, 2, 2, 3, 3, 4]
    zar2 = [1, 3, 5, 7, 9, 11]
    import random
    random.seed(7)
    for _ in range(n):
        yield random.choice(zar1) + random.choice(zar2)

def p95_necklace_sayisi(n, k=2):
    """Kolyeler (necklaces): n boncuk k renk, dönmeye eşdeğer
    Armoni: Devri modlar → modal eşdeğerlik"""
    def necklace(n, k):
        from math import gcd
        total = sum(k ** gcd(i, n) for i in range(n))
        return total // n
    for i in range(1, n + 1):
        yield necklace(i, k)

def p96_wallis_kismi(n):
    """Wallis çarpımı kısmi çarpımları: π/2 yaklaşımı
    Armoni: Yavaş π yaklaşımı → uzun vadeli çözüm hissi"""
    product = 1.0
    for i in range(1, n + 1):
        product *= (2 * i) * (2 * i) / ((2 * i - 1) * (2 * i + 1))
        yield product

def p97_continued_fraction_e(n):
    """e sayısının sürekli kesir yaklaşımları
    Armoni: e ≈ 2.718... → irrasyonel oran, hiç tekrar etmez"""
    # e = [2; 1,2,1,1,4,1,1,6,1,1,8,...]
    def cf_e(idx):
        if idx == 0: return 2
        if idx % 3 == 2: return 2 * (idx // 3 + 1)
        return 1
    h_prev, h_curr = 1, cf_e(0)
    k_prev, k_curr = 0, 1
    yield h_curr / k_curr
    for i in range(1, n):
        a = cf_e(i)
        h_prev, h_curr = h_curr, a * h_curr + h_prev
        k_prev, k_curr = k_curr, a * k_curr + k_prev
        yield h_curr / k_curr

def p98_langton_karinci(n, grid_size=20):
    """Langton'un karıncası: aktif hücre koordinat toplamı
    Armoni: Kaotikten düzene → müzikal form gelişimi"""
    grid = set()
    x, y = grid_size // 2, grid_size // 2
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    d = 0
    for _ in range(n):
        yield abs(x - grid_size // 2) + abs(y - grid_size // 2)
        if (x, y) in grid:
            grid.remove((x, y))
            d = (d - 1) % 4
        else:
            grid.add((x, y))
            d = (d + 1) % 4
        dx, dy = directions[d]
        x, y = x + dx, y + dy

def p99_tupper_formulü(n):
    """Tupper'ın kendi kendini referans eden formülü çıktısı
    Armoni: Matematiksel öz-referans → meta-melodi"""
    # Basitleştirilmiş Tupper piksel değerleri
    k = 960939379918958884971672962127852754715004339660129306651505519271702802395266424689642842174350718121267153782770623355993237280874144307891325963941337723487857735749823926629715517173716995165232890538221612403238855866184013235585136048828693337902491454229288667081096184496091705183454067827731551705405381627380967602565625016981482083418783163849115590225610003652351370343874461848378737238198224849863465033159410054974700593138339226497249461751545728366702369745461014655997933798537483143786841806593422227898388722980000748404719
    pixels = []
    for j in range(17, -1, -1):
        row = []
        for i in range(106):
            val = k + j
            bit = (val // (17 * i + 1)) % 2
            row.append(int(bit))
        pixels.extend(row[:n])
    yield from pixels[:n]

def p100_selbst_referenz(n):
    """Kendi kendini tanımlayan dizi: a(n) = n. karakteri olan sayı
    Armoni: Dil ve müzik arasındaki köprü → meta-kompozisyon"""
    # Gödel benzeri öz-referans: "Bu dizinin n. elemanı n'dir"
    # Basitleştirilmiş: her sayı kendi indeksini içerir
    for i in range(n):
        # Kaç tane i değeri şimdiye kadar üretildi?
        yield i  # Öz-referanslı temel: a(i) = i


# ─────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR VE DEMO
# ─────────────────────────────────────────────

def normalize(seq, low=0.0, high=1.0):
    """Diziyi [low, high] aralığına normalize et"""
    values = list(seq)
    mn, mx = min(values), max(values)
    if mx == mn:
        return [low] * len(values)
    return [low + (v - mn) / (mx - mn) * (high - low) for v in values]

def to_frekans(seq, temel=220.0, oktav=1):
    """Normalize edilmiş diziyi frekansa çevir
    temel: A3 = 220 Hz (ya da istediğin temel)
    oktav: kaç oktav aralık kullanılacak"""
    values = list(seq)
    mn, mx = min(values), max(values)
    if mx == mn:
        return [temel] * len(values)
    return [temel * (2 ** (oktav * (v - mn) / (mx - mn))) for v in values]

def to_semitone(seq, mod=12):
    """Diziyi yarım ton aralıklarına (0-11) eşle"""
    values = list(seq)
    mn, mx = min(values), max(values)
    if mx == mn:
        return [0] * len(values)
    return [round((v - mn) / (mx - mn) * (mod - 1)) for v in values]

def ornuntu_listesi():
    """Tüm örüntü fonksiyonlarını listele"""
    oruntular = {
        "p01": ("Fibonacci", "Klasik Sayısal"),
        "p02": ("Lucas", "Klasik Sayısal"),
        "p03": ("Tribonacci", "Klasik Sayısal"),
        "p04": ("Padovan", "Klasik Sayısal"),
        "p05": ("Pell (√2 oranı)", "Klasik Sayısal"),
        "p06": ("Jacobsthal", "Klasik Sayısal"),
        "p07": ("Perrin", "Klasik Sayısal"),
        "p08": ("Stern-Brocot", "Klasik Sayısal"),
        "p09": ("Sylvester (üstel)", "Klasik Sayısal"),
        "p10": ("Catalan", "Klasik Sayısal"),
        "p11": ("Asal Sayılar", "Asal/Mod"),
        "p12": ("Asal Boşluklar", "Asal/Mod"),
        "p13": ("Fibonacci mod 12", "Asal/Mod"),
        "p14": ("Geometrik mod 7", "Asal/Mod"),
        "p15": ("Collatz", "Asal/Mod"),
        "p16": ("Euler Totient", "Asal/Mod"),
        "p17": ("Şanslı Sayılar", "Asal/Mod"),
        "p18": ("Golomb", "Asal/Mod"),
        "p19": ("Recamán", "Asal/Mod"),
        "p20": ("Kaprekar 6174", "Asal/Mod"),
        "p21": ("Harmonik Seri", "Geometrik/Trig"),
        "p22": ("Overtone Serisi", "Geometrik/Trig"),
        "p23": ("Just Intonation", "Geometrik/Trig"),
        "p24": ("Pisagor Skala", "Geometrik/Trig"),
        "p25": ("Sinüs Harmonikleri", "Geometrik/Trig"),
        "p26": ("Lissajous X", "Geometrik/Trig"),
        "p27": ("Lissajous Y", "Geometrik/Trig"),
        "p28": ("Chebyshev", "Geometrik/Trig"),
        "p29": ("Fourier Kare Dalga", "Geometrik/Trig"),
        "p30": ("Titreşim (Beat)", "Geometrik/Trig"),
        "p31": ("Lojistik Harita", "Fraktal/Kaotik"),
        "p32": ("Tent Harita", "Fraktal/Kaotik"),
        "p33": ("Lorenz X", "Fraktal/Kaotik"),
        "p34": ("Hénon X", "Fraktal/Kaotik"),
        "p35": ("Arnold'un Kedisi", "Fraktal/Kaotik"),
        "p36": ("Sierpinski Toplamı", "Fraktal/Kaotik"),
        "p37": ("Dragon Eğrisi", "Fraktal/Kaotik"),
        "p38": ("Thue-Morse", "Fraktal/Kaotik"),
        "p39": ("Cantor Seti", "Fraktal/Kaotik"),
        "p40": ("Mandelbrot İter.", "Fraktal/Kaotik"),
        "p41": ("Beşli Çember", "Müzikal Teori"),
        "p42": ("Altın Oran Melodi", "Müzikal Teori"),
        "p43": ("12-TET Aralıklar", "Müzikal Teori"),
        "p44": ("Modal Skala", "Müzikal Teori"),
        "p45": ("Pentatonik", "Müzikal Teori"),
        "p46": ("41-TET Mikrotonal", "Müzikal Teori"),
        "p47": ("Öklid Ritmi", "Müzikal Teori"),
        "p48": ("Zarb Ritmi (3+3+2)", "Müzikal Teori"),
        "p49": ("Φ Ritim", "Müzikal Teori"),
        "p50": ("12-Ton Seri", "Müzikal Teori"),
        "p51": ("Pascal Üçgeni", "Kombinatorik"),
        "p52": ("Motzkin", "Kombinatorik"),
        "p53": ("Narayana", "Kombinatorik"),
        "p54": ("Stirling 1. tür", "Kombinatorik"),
        "p55": ("Bell Sayıları", "Kombinatorik"),
        "p56": ("Gray Kodu", "Kombinatorik"),
        "p57": ("Rudin-Shapiro", "Kombinatorik"),
        "p58": ("Baum-Sweet", "Kombinatorik"),
        "p59": ("Kolakoski", "Kombinatorik"),
        "p60": ("Lempel-Ziv", "Kombinatorik"),
        "p61": ("Kuantum Harmonik", "Fiziksel/Dalga"),
        "p62": ("Dalga Paketi", "Fiziksel/Dalga"),
        "p63": ("Tam Dalga Doğr.", "Fiziksel/Dalga"),
        "p64": ("FM Sentezi", "Fiziksel/Dalga"),
        "p65": ("AM Sentezi", "Fiziksel/Dalga"),
        "p66": ("Duffing Osilatör", "Fiziksel/Dalga"),
        "p67": ("Van der Pol", "Fiziksel/Dalga"),
        "p68": ("Kuramoto Modeli", "Fiziksel/Dalga"),
        "p69": ("Gecikmeli Geri Bes.", "Fiziksel/Dalga"),
        "p70": ("Haar Wavelet", "Fiziksel/Dalga"),
        "p71": ("Mersenne Eşik", "Sayılar Teorisi"),
        "p72": ("Mükemmel Sayılar", "Sayılar Teorisi"),
        "p73": ("Liouville λ", "Sayılar Teorisi"),
        "p74": ("Möbius μ", "Sayılar Teorisi"),
        "p75": ("Sigma Bölücüsü", "Sayılar Teorisi"),
        "p76": ("Pisagor Üçlüleri", "Sayılar Teorisi"),
        "p77": ("Babilon Base-60", "Sayılar Teorisi"),
        "p78": ("Mısır Kesirleri", "Sayılar Teorisi"),
        "p79": ("Zeckendorf", "Sayılar Teorisi"),
        "p80": ("Farey Dizisi", "Sayılar Teorisi"),
        "p81": ("Kural 30", "Automata"),
        "p82": ("Kural 90", "Automata"),
        "p83": ("Game of Life", "Automata"),
        "p84": ("Sandpile Model", "Automata"),
        "p85": ("XOR-shift PRNG", "Automata"),
        "p86": ("Hilbert Eğrisi X", "İleri Örüntüler"),
        "p87": ("Vicsek Fraktal", "İleri Örüntüler"),
        "p88": ("Voronoi Mesafe", "İleri Örüntüler"),
        "p89": ("L-Sistemi Uzunluk", "İleri Örüntüler"),
        "p90": ("Kvaziperiyodik 1D", "İleri Örüntüler"),
        "p91": ("Asal Spirali Açı", "İleri Örüntüler"),
        "p92": ("Hofstadter Q", "İleri Örüntüler"),
        "p93": ("Hofstadter G", "İleri Örüntüler"),
        "p94": ("Sicherman Zar", "İleri Örüntüler"),
        "p95": ("Kolye Sayısı", "İleri Örüntüler"),
        "p96": ("Wallis Çarpımı", "İleri Örüntüler"),
        "p97": ("e Sürekli Kesir", "İleri Örüntüler"),
        "p98": ("Langton Karıncası", "İleri Örüntüler"),
        "p99": ("Tupper Formülü", "İleri Örüntüler"),
        "p100": ("Öz-Referans", "İleri Örüntüler"),
    }
    return oruntular


# ─────────────────────────────────────────────
# 100 ÖRÜNTÜ ÇIKTILARINI DOSYAYA YAZ (logic test için)
# ─────────────────────────────────────────────

def _serialize_value(v):
    """Örüntü çıktı değerini CSV'ye yazılabilir forma çevirir."""
    from fractions import Fraction
    if isinstance(v, Fraction):
        return float(v)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v
    if hasattr(v, "real") and hasattr(v, "imag"):  # complex
        return f"{v.real:.6f}+{v.imag:.6f}j"
    return str(v)


def export_mathematical_pattern_outputs(n: int = 12, out_dir: Path | None = None) -> None:
    """
    100 matematiksel örüntü fonksiyonunu çalıştırır ve çıktıları CSV dosyalarına yazar.
    Proje kökünden:  python scripts/brute_force_patterns.py --export-patterns

    Üretilen dosyalar (out_dir = results/stats varsayılan):
      - mathematical_pattern_outputs_long.csv  (pattern_id, pattern_name, category, index, value)
      - mathematical_pattern_outputs_wide.csv (pattern_id, pattern_name, category, v0, v1, ...)

    Daha uzun dizi icin: export_mathematical_pattern_outputs(n=24)
    """
    import inspect
    import re

    out_dir = out_dir or (Path(__file__).resolve().parent.parent / "results" / "stats")
    out_dir.mkdir(parents=True, exist_ok=True)

    oruntular = ornuntu_listesi()
    # globals() içinde p01_fibonacci, p02_lucas, ... bul
    this_module = __import__(__name__)
    g = globals()
    func_map = {}
    for name, obj in g.items():
        if callable(obj) and re.match(r"^p\d{2}_", name):
            pid = name[:3]  # p01, p02, ...
            if pid in oruntular and pid not in func_map:
                func_map[pid] = obj

    long_rows = []
    wide_rows = []

    for pattern_id in sorted(oruntular.keys(), key=lambda x: int(x[1:])):
        if pattern_id not in func_map:
            continue
        func = func_map[pattern_id]
        pattern_name, category = oruntular[pattern_id]
        try:
            sig = inspect.signature(func)
            params = {"n": n}
            for pname, p in sig.parameters.items():
                if pname == "n":
                    continue
                if p.default is not inspect.Parameter.empty:
                    params[pname] = p.default
            gen = func(**params)
            values = []
            for _ in range(n):
                try:
                    v = next(gen)
                    values.append(_serialize_value(v))
                except StopIteration:
                    break
        except Exception as e:
            values = [f"ERROR:{e!s}"]

        for i, val in enumerate(values):
            long_rows.append({
                "pattern_id": pattern_id,
                "pattern_name": pattern_name,
                "category": category,
                "index": i,
                "value": val,
            })
        wide_row = {"pattern_id": pattern_id, "pattern_name": pattern_name, "category": category}
        for i, val in enumerate(values):
            wide_row[f"v{i}"] = val
        wide_rows.append(wide_row)

    long_path = out_dir / "mathematical_pattern_outputs_long.csv"
    wide_path = out_dir / "mathematical_pattern_outputs_wide.csv"
    pd.DataFrame(long_rows).to_csv(long_path, index=False)
    pd.DataFrame(wide_rows).to_csv(wide_path, index=False)
    print(f"100 matematiksel oruntu ciktilari yazildi:")
    print(f"  - {long_path}")
    print(f"  - {wide_path}")


# ─────────────────────────────────────────────
# DEMO: BİRKAÇ ÖRNEĞİ ÇALIŞTIR
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--export-patterns":
        n_val = 12
        if len(sys.argv) > 2:
            try:
                n_val = int(sys.argv[2])
            except ValueError:
                pass
        export_mathematical_pattern_outputs(n=n_val)
    else:
        main() 