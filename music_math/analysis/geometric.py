"""
Geometric Pattern Visualizer - Müziksel Geometrik Pattern'ler

Müzikteki geometrik yapıları tespit eder ve görselleştirir:
- Spiral patterns (Fibonacci, Archimedean)
- Mandala structures (dairesel simetri)
- Circular representations (tonnetz, pitch class circle)
- Geometric shapes (üçgen, kare, beşgen)
- Star polygons (yıldız çokgenler)

Kullanım:
    from music_math.analysis.geometric import (
        detect_spiral_pattern,
        create_mandala_structure,
        build_circular_representation,
        analyze_geometric_properties,
    )
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import math

from music_math.core.types import NoteEvent


PHI = (1 + math.sqrt(5)) / 2  # Altın oran


class GeometricShape(Enum):
    """Geometrik şekil tipleri."""
    TRIANGLE = "triangle"  # Üçgen
    SQUARE = "square"  # Kare
    PENTAGON = "pentagon"  # Beşgen
    HEXAGON = "hexagon"  # Altıgen
    OCTAGON = "octagon"  # Sekizgen
    STAR_TRIANGLE = "star_triangle"  # Üçgen yıldız
    STAR_PENTAGON = "star_pentagon"  # Beşgen yıldız (pentagram)
    SPIRAL = "spiral"  # Spiral
    MANDALA = "mandala"  # Mandala


@dataclass
class GeometricPoint:
    """2D geometrik nokta."""
    x: float
    y: float
    pitch: Optional[int] = None
    time: Optional[float] = None


@dataclass
class GeometricPattern:
    """Bulunan geometrik pattern."""
    shape_type: GeometricShape
    vertices: List[GeometricPoint]
    center: GeometricPoint
    radius: float
    rotation: float
    confidence: float
    notes_involved: List[int]  # Nota indeksleri


@dataclass
class SpiralPattern:
    """Spiral pattern."""
    center: GeometricPoint
    start_radius: float
    end_radius: float
    num_turns: float
    direction: str  # "clockwise", "counterclockwise"
    growth_rate: float  # Fibonacci vs Archimedean
    confidence: float


@dataclass
class MandalaLayer:
    """Mandala katmanı."""
    radius: float
    num_points: int
    points: List[GeometricPoint]
    symmetry_order: int


@dataclass
class GeometricAnalysis:
    """Geometrik analiz sonuçları."""
    shapes: List[GeometricPattern]
    spirals: List[SpiralPattern]
    mandala_layers: List[MandalaLayer]
    circular_coords: List[GeometricPoint]
    geometric_complexity: float
    symmetry_orders: List[int]


def pitch_to_circular_coordinates(
    pitch: int,
    radius: float = 1.0,
    center: Tuple[float, float] = (0.0, 0.0)
) -> GeometricPoint:
    """
    Pitch'i dairesel koordinatlara dönüştürür (pitch class circle).
    
    Args:
        pitch: MIDI pitch (0-127)
        radius: Daire yarıçapı
        center: Merkez noktası (x, y)
        
    Returns:
        GeometricPoint
    """
    # Pitch class (0-11)
    pc = pitch % 12
    
    # Açı: C=0, C#=30, D=60, ... (saat yönünün tersine)
    angle = (pc / 12) * 2 * math.pi - math.pi / 2  # -90 derece offset (C yukarıda)
    
    x = center[0] + radius * math.cos(angle)
    y = center[1] + radius * math.sin(angle)
    
    return GeometricPoint(x, y, pitch)


def sequence_to_spiral_coordinates(
    pitches: List[int],
    spiral_type: str = "fibonacci",
    center: Tuple[float, float] = (0.0, 0.0)
) -> List[GeometricPoint]:
    """
    Pitch dizisini spiral koordinatlara dönüştürür.
    
    Args:
        pitches: Pitch değerleri
        spiral_type: "fibonacci" veya "archimedean"
        center: Merkez
        
    Returns:
        GeometricPoint listesi
    """
    points = []
    n = len(pitches)
    
    if spiral_type == "fibonacci":
        # Fibonacci spiral: r = a * phi^(theta)
        a = 0.1
        for i, pitch in enumerate(pitches):
            theta = i * 0.5  # Açı artsın
            r = a * (PHI ** (theta / (2 * math.pi)))
            x = center[0] + r * math.cos(theta)
            y = center[1] + r * math.sin(theta)
            points.append(GeometricPoint(x, y, pitch))
    
    else:  # archimedean
        # Archimedean spiral: r = a + b * theta
        a, b = 0.1, 0.05
        for i, pitch in enumerate(pitches):
            theta = i * 0.3
            r = a + b * theta
            x = center[0] + r * math.cos(theta)
            y = center[1] + r * math.sin(theta)
            points.append(GeometricPoint(x, y, pitch))
    
    return points


def detect_spiral_pattern(
    pitches: List[int],
    tolerance: float = 0.15
) -> Optional[SpiralPattern]:
    """
    Spiral pattern tespiti.
    
    Args:
        pitches: Pitch dizisi
        tolerance: Tolerans
        
    Returns:
        SpiralPattern veya None
    """
    if len(pitches) < 10:
        return None
    
    # Dairesel koordinatlara dönüştür
    points = [pitch_to_circular_coordinates(p, radius=1.0) for p in pitches]
    
    # Merkez hesapla
    center_x = np.mean([p.x for p in points])
    center_y = np.mean([p.y for p in points])
    center = GeometricPoint(center_x, center_y)
    
    # Her noktanın merkeze uzaklığı ve açısı
    radii = []
    angles = []
    
    for p in points:
        dx = p.x - center_x
        dy = p.y - center_y
        r = math.sqrt(dx**2 + dy**2)
        theta = math.atan2(dy, dx)
        radii.append(r)
        angles.append(theta)
    
    # Spiral kontrolü: yarıçap sürekli artmalı
    increasing_count = sum(1 for i in range(len(radii) - 1) if radii[i+1] > radii[i])
    increasing_ratio = increasing_count / max(1, len(radii) - 1)
    
    if increasing_ratio < 0.7:  # %70'inden azı artıyorsa spiral değil
        return None
    
    # Fibonacci spiral testi: r ~ phi^(theta)
    log_radii = [math.log(r) if r > 0 else 0 for r in radii]
    
    # Linear regression: log(r) vs theta
    if len(angles) > 1 and len(set(angles)) > 1:
        correlation = np.corrcoef(angles, log_radii)[0, 1]
        is_fibonacci = abs(correlation) > 0.7
    else:
        is_fibonacci = False
    
    # Dönüş sayısı
    angle_diff = abs(angles[-1] - angles[0])
    num_turns = angle_diff / (2 * math.pi)
    
    growth_rate = PHI if is_fibonacci else 1.1
    
    return SpiralPattern(
        center=center,
        start_radius=radii[0],
        end_radius=radii[-1],
        num_turns=num_turns,
        direction="clockwise" if angles[-1] > angles[0] else "counterclockwise",
        growth_rate=growth_rate,
        confidence=min(1.0, increasing_ratio)
    )


def create_mandala_structure(
    pitches: List[int],
    num_layers: int = 3
) -> List[MandalaLayer]:
    """
    Mandala yapısı oluşturur (dairesel simetri).
    
    Args:
        pitches: Pitch değerleri
        num_layers: Katman sayısı
        
    Returns:
        MandalaLayer listesi
    """
    if len(pitches) < 12:
        return []
    
    layers = []
    max_radius = 100
    
    for layer_idx in range(num_layers):
        radius = max_radius * (layer_idx + 1) / num_layers
        
        # Bu katman için pitch'leri seç
        start_idx = layer_idx * (len(pitches) // num_layers)
        end_idx = min(start_idx + 12, len(pitches))
        layer_pitches = pitches[start_idx:end_idx]
        
        num_points = len(layer_pitches)
        if num_points == 0:
            continue
        
        # Daireye yerleştir
        points = []
        for i, pitch in enumerate(layer_pitches):
            angle = (i / num_points) * 2 * math.pi - math.pi / 2
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            points.append(GeometricPoint(x, y, pitch))
        
        # Simetri derecesi (en küçük simetri)
        symmetry_order = num_points
        for div in range(2, num_points // 2 + 1):
            if num_points % div == 0:
                symmetry_order = div
                break
        
        layers.append(MandalaLayer(
            radius=radius,
            num_points=num_points,
            points=points,
            symmetry_order=symmetry_order
        ))
    
    return layers


def detect_geometric_shapes(
    pitches: List[int],
    tolerance: float = 0.1
) -> List[GeometricPattern]:
    """
    Geometrik şekilleri tespit eder (üçgen, kare, beşgen, vb.).
    
    Args:
        pitches: Pitch değerleri
        tolerance: Tolerans
        
    Returns:
        GeometricPattern listesi
    """
    shapes = []
    n = len(pitches)
    
    # Her ardışık grup için şekil kontrolü
    for num_vertices in [3, 4, 5, 6, 8]:  # Üçgen, kare, beşgen, altıgen, sekizgen
        for start_idx in range(n - num_vertices + 1):
            window = pitches[start_idx:start_idx + num_vertices]
            
            # Dairesel koordinatlar
            points = [pitch_to_circular_coordinates(p, radius=100) for p in window]
            
            # Merkez
            center_x = np.mean([p.x for p in points])
            center_y = np.mean([p.y for p in points])
            center = GeometricPoint(center_x, center_y)
            
            # Köşelerin merkeze uzaklıkları
            radii = [math.sqrt((p.x - center_x)**2 + (p.y - center_y)**2) for p in points]
            radius_mean = np.mean(radii)
            radius_std = np.std(radii)
            
            # Düzgün çokgen: tüm köşeler yaklaşık aynı uzaklıkta
            if radius_std / radius_mean < tolerance:
                # Açılar arası mesafe eşit mi?
                angles = [math.atan2(p.y - center_y, p.x - center_x) for p in points]
                angle_diffs = []
                for i in range(len(angles)):
                    diff = abs(angles[(i+1) % len(angles)] - angles[i])
                    if diff > math.pi:
                        diff = 2 * math.pi - diff
                    angle_diffs.append(diff)
                
                expected_angle = 2 * math.pi / num_vertices
                angle_variance = np.var([(d - expected_angle) / expected_angle for d in angle_diffs])
                
                if angle_variance < tolerance:
                    # Şekil tipini belirle
                    shape_type_map = {
                        3: GeometricShape.TRIANGLE,
                        4: GeometricShape.SQUARE,
                        5: GeometricShape.PENTAGON,
                        6: GeometricShape.HEXAGON,
                        8: GeometricShape.OCTAGON,
                    }
                    
                    confidence = 1.0 - (radius_std / radius_mean + angle_variance) / 2
                    
                    shapes.append(GeometricPattern(
                        shape_type=shape_type_map.get(num_vertices, GeometricShape.TRIANGLE),
                        vertices=points,
                        center=center,
                        radius=radius_mean,
                        rotation=angles[0],
                        confidence=round(confidence, 3),
                        notes_involved=list(range(start_idx, start_idx + num_vertices))
                    ))
    
    # En iyi şekilleri seç (çakışmayan)
    shapes.sort(key=lambda s: s.confidence, reverse=True)
    non_overlapping = []
    used_indices = set()
    
    for shape in shapes:
        overlaps = any(idx in used_indices for idx in shape.notes_involved)
        if not overlaps:
            non_overlapping.append(shape)
            used_indices.update(shape.notes_involved)
    
    return non_overlapping[:10]


def detect_star_polygons(
    pitches: List[int],
    tolerance: float = 0.15
) -> List[GeometricPattern]:
    """
    Yıldız çokgenlerini tespit eder (pentagram vb.).
    
    Args:
        pitches: Pitch değerleri
        tolerance: Tolerans
        
    Returns:
        GeometricPattern listesi
    """
    stars = []
    n = len(pitches)
    
    # Pentagram (5-pointed star): 5 köşe, atlama = 2
    # Star of David (6-pointed): 6 köşe, atlama = 2
    star_configs = [
        (5, 2, GeometricShape.STAR_PENTAGON),
        (6, 2, GeometricShape.STAR_TRIANGLE),
    ]
    
    for num_vertices, skip, shape_type in star_configs:
        for start_idx in range(n - num_vertices + 1):
            window = pitches[start_idx:start_idx + num_vertices]
            
            # Yıldız sıralaması: her skip eleman atlayarak
            star_order = [(i * skip) % num_vertices for i in range(num_vertices)]
            star_pitches = [window[i] for i in star_order]
            
            # Kontrol: yıldız geçerli mi?
            points = [pitch_to_circular_coordinates(p, radius=100) for p in star_pitches]
            
            # Düzgünlük kontrolü
            center_x = np.mean([p.x for p in points])
            center_y = np.mean([p.y for p in points])
            radii = [math.sqrt((p.x - center_x)**2 + (p.y - center_y)**2) for p in points]
            
            if np.std(radii) / np.mean(radii) < tolerance:
                confidence = 1.0 - np.std(radii) / np.mean(radii)
                
                stars.append(GeometricPattern(
                    shape_type=shape_type,
                    vertices=points,
                    center=GeometricPoint(center_x, center_y),
                    radius=np.mean(radii),
                    rotation=0,
                    confidence=round(confidence, 3),
                    notes_involved=list(range(start_idx, start_idx + num_vertices))
                ))
    
    return stars[:5]


def calculate_geometric_complexity(
    shapes: List[GeometricPattern],
    spirals: List[SpiralPattern],
    mandala_layers: List[MandalaLayer]
) -> float:
    """
    Geometrik karmaşıklık skoru.
    
    Args:
        shapes: Şekiller
        spirals: Spiraller
        mandala_layers: Mandala katmanları
        
    Returns:
        Karmaşıklık [0, 1]
    """
    score = 0.0
    
    # Şekil çeşitliliği
    unique_shape_types = len(set(s.shape_type for s in shapes))
    score += unique_shape_types * 0.1
    
    # Spiral varlığı
    if spirals:
        score += 0.2 * len(spirals)
    
    # Mandala katmanları
    score += len(mandala_layers) * 0.1
    
    # Toplam şekil sayısı
    score += len(shapes) * 0.05
    
    return min(1.0, score)


def analyze_geometric_properties(events: List[NoteEvent]) -> GeometricAnalysis:
    """
    Kapsamlı geometrik analiz.
    
    Args:
        events: Nota olayları
        
    Returns:
        GeometricAnalysis sonuçları
    """
    pitches = [e.pitch for e in events]
    
    if len(pitches) < 6:
        return GeometricAnalysis(
            shapes=[],
            spirals=[],
            mandala_layers=[],
            circular_coords=[],
            geometric_complexity=0.0,
            symmetry_orders=[]
        )
    
    # Dairesel koordinatlar
    circular_coords = [pitch_to_circular_coordinates(p) for p in pitches]
    
    # Spiraller
    spiral = detect_spiral_pattern(pitches)
    spirals = [spiral] if spiral else []
    
    # Mandala
    mandala_layers = create_mandala_structure(pitches, num_layers=3)
    
    # Şekiller
    shapes = detect_geometric_shapes(pitches)
    
    # Yıldızlar
    stars = detect_star_polygons(pitches)
    shapes.extend(stars)
    
    # Karmaşıklık
    complexity = calculate_geometric_complexity(shapes, spirals, mandala_layers)
    
    # Simetri dereceleri
    symmetry_orders = list(set(layer.symmetry_order for layer in mandala_layers))
    
    return GeometricAnalysis(
        shapes=shapes,
        spirals=spirals,
        mandala_layers=mandala_layers,
        circular_coords=circular_coords,
        geometric_complexity=round(complexity, 3),
        symmetry_orders=symmetry_orders
    )


def extract_geometric_features(events: List[NoteEvent]) -> Dict[str, float]:
    """
    Feature extraction için geometrik özellikler.
    
    Args:
        events: Nota olayları
        
    Returns:
        Feature sözlüğü
    """
    analysis = analyze_geometric_properties(events)
    
    features = {
        "geometric_complexity": analysis.geometric_complexity,
        "geometric_shape_count": len(analysis.shapes),
        "geometric_spiral_count": len(analysis.spirals),
        "geometric_mandala_layers": len(analysis.mandala_layers),
        "has_spiral": 1.0 if analysis.spirals else 0.0,
        "has_mandala": 1.0 if analysis.mandala_layers else 0.0,
    }
    
    # Şekil tipleri
    shape_counts = {}
    for shape in analysis.shapes:
        shape_type = shape.shape_type.value
        shape_counts[f"shape_{shape_type}"] = shape_counts.get(f"shape_{shape_type}", 0) + 1
    
    for shape_type, count in shape_counts.items():
        features[f"geometric_{shape_type}_count"] = count
    
    # Simetri
    if analysis.symmetry_orders:
        features["max_symmetry_order"] = max(analysis.symmetry_orders)
        features["avg_symmetry_order"] = np.mean(analysis.symmetry_orders)
    else:
        features["max_symmetry_order"] = 0
        features["avg_symmetry_order"] = 0
    
    return features


__all__ = [
    "GeometricShape",
    "GeometricPoint",
    "GeometricPattern",
    "SpiralPattern",
    "MandalaLayer",
    "GeometricAnalysis",
    "PHI",
    "pitch_to_circular_coordinates",
    "sequence_to_spiral_coordinates",
    "detect_spiral_pattern",
    "create_mandala_structure",
    "detect_geometric_shapes",
    "detect_star_polygons",
    "calculate_geometric_complexity",
    "analyze_geometric_properties",
    "extract_geometric_features",
]
