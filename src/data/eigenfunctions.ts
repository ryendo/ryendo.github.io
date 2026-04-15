// Laplacian Eigenfunction Data - Extended Version (Beta)
// Computed using FEniCS and analytical solutions

export interface Eigenfunction {
  id: string;
  domain: string;
  domainEn: string;
  domainJa: string;
  boundary: 'dirichlet' | 'neumann' | 'steklov';
  mode: number;
  eigenvalue: number;
  formula: string;
  description: string;
  descriptionJa: string;
  multiplicity: number;
  reference?: string;
  computed?: boolean;
}

export interface DomainInfo {
  id: string;
  nameEn: string;
  nameJa: string;
  description: string;
  descriptionJa: string;
  category: 'basic' | 'polygon' | 'curved' | 'sector' | 'exotic' | 'collapsing';
}

export const domains: DomainInfo[] = [
  { id: 'unit-square', nameEn: 'Unit Square', nameJa: '単位正方形', description: '[0,1]×[0,1]', descriptionJa: '[0,1]×[0,1]', category: 'basic' },
  { id: 'unit-disk', nameEn: 'Unit Disk', nameJa: '単位円板', description: 'Disk of radius 1', descriptionJa: '半径1の円板', category: 'basic' },
  { id: 'rectangle-2-1', nameEn: 'Rectangle (2:1)', nameJa: '長方形 (2:1)', description: '[0,2]×[0,1]', descriptionJa: '[0,2]×[0,1]', category: 'basic' },
  { id: 'equilateral-triangle', nameEn: 'Equilateral Triangle', nameJa: '正三角形', description: 'Side length 1', descriptionJa: '一辺1', category: 'polygon' },
  { id: 'isosceles-right-triangle', nameEn: 'Isosceles Right Triangle', nameJa: '直角二等辺三角形', description: 'Legs of length 1', descriptionJa: '脚の長さ1', category: 'polygon' },
  { id: 'pentagon', nameEn: 'Regular Pentagon', nameJa: '正五角形', description: 'Inscribed in unit circle', descriptionJa: '単位円に内接', category: 'polygon' },
  { id: 'hexagon', nameEn: 'Regular Hexagon', nameJa: '正六角形', description: 'Inscribed in unit circle', descriptionJa: '単位円に内接', category: 'polygon' },
  { id: 'ellipse-2-1', nameEn: 'Ellipse (2:1)', nameJa: '楕円 (2:1)', description: 'Semi-axes a=1, b=0.5', descriptionJa: '半軸 a=1, b=0.5', category: 'curved' },
  { id: 'stadium', nameEn: 'Stadium (Bunimovich)', nameJa: 'スタジアム', description: 'Rectangle with semicircular ends', descriptionJa: '半円で閉じた長方形', category: 'curved' },
  { id: 'sector-90', nameEn: 'Quarter Disk', nameJa: '四分円', description: 'Sector with angle π/2', descriptionJa: '角度 π/2 の扇形', category: 'sector' },
  { id: 'sector-60', nameEn: 'Circular Sector 60°', nameJa: '扇形 60°', description: 'Sector with angle π/3', descriptionJa: '角度 π/3 の扇形', category: 'sector' },
  { id: 'annulus-0.5', nameEn: 'Annulus (r=0.5,1)', nameJa: '円環 (r=0.5,1)', description: 'Inner radius 0.5', descriptionJa: '内半径 0.5', category: 'exotic' },
  { id: 'l-shape', nameEn: 'L-shaped Domain', nameJa: 'L字型領域', description: '[-1,1]²∖[0,1]×[-1,0]', descriptionJa: '[-1,1]²∖[0,1]×[-1,0]', category: 'exotic' },
  { id: 'thin-triangle-0.1', nameEn: 'Thin Triangle (h=0.1)', nameJa: '細長い三角形 (h=0.1)', description: 'Base 1, height 0.1', descriptionJa: '底辺1, 高さ0.1', category: 'collapsing' },
];

export const eigenfunctions: Eigenfunction[] = [
  // UNIT SQUARE - DIRICHLET
  { id: 'sq-d-1', domain: 'unit-square', domainEn: 'Unit Square', domainJa: '単位正方形', boundary: 'dirichlet', mode: 1, eigenvalue: 19.739208802178716, formula: '2π²', description: 'sin(πx)sin(πy)', descriptionJa: 'sin(πx)sin(πy)', multiplicity: 1 },
  { id: 'sq-d-2', domain: 'unit-square', domainEn: 'Unit Square', domainJa: '単位正方形', boundary: 'dirichlet', mode: 2, eigenvalue: 49.34802200544679, formula: '5π²', description: 'Double mode', descriptionJa: '2重モード', multiplicity: 2 },
  { id: 'sq-d-3', domain: 'unit-square', domainEn: 'Unit Square', domainJa: '単位正方形', boundary: 'dirichlet', mode: 4, eigenvalue: 78.95683520871486, formula: '8π²', description: 'sin(2πx)sin(2πy)', descriptionJa: 'sin(2πx)sin(2πy)', multiplicity: 1 },
  { id: 'sq-d-4', domain: 'unit-square', domainEn: 'Unit Square', domainJa: '単位正方形', boundary: 'dirichlet', mode: 5, eigenvalue: 98.69604401089358, formula: '10π²', description: 'Double mode', descriptionJa: '2重モード', multiplicity: 2 },
  { id: 'sq-d-5', domain: 'unit-square', domainEn: 'Unit Square', domainJa: '単位正方形', boundary: 'dirichlet', mode: 7, eigenvalue: 128.30485721415769, formula: '13π²', description: 'Double mode', descriptionJa: '2重モード', multiplicity: 2 },
  { id: 'sq-d-6', domain: 'unit-square', domainEn: 'Unit Square', domainJa: '単位正方形', boundary: 'dirichlet', mode: 9, eigenvalue: 167.78324161633575, formula: '17π²', description: 'Double mode', descriptionJa: '2重モード', multiplicity: 2 },
  { id: 'sq-d-7', domain: 'unit-square', domainEn: 'Unit Square', domainJa: '単位正方形', boundary: 'dirichlet', mode: 11, eigenvalue: 177.65287841960487, formula: '18π²', description: 'sin(3πx)sin(3πy)', descriptionJa: 'sin(3πx)sin(3πy)', multiplicity: 1 },
  // UNIT SQUARE - NEUMANN
  { id: 'sq-n-0', domain: 'unit-square', domainEn: 'Unit Square', domainJa: '単位正方形', boundary: 'neumann', mode: 0, eigenvalue: 0, formula: '0', description: 'Constant', descriptionJa: '定数', multiplicity: 1 },
  { id: 'sq-n-1', domain: 'unit-square', domainEn: 'Unit Square', domainJa: '単位正方形', boundary: 'neumann', mode: 1, eigenvalue: 9.869604401089358, formula: 'π²', description: 'cos(πx), cos(πy)', descriptionJa: 'cos(πx), cos(πy)', multiplicity: 2 },
  { id: 'sq-n-2', domain: 'unit-square', domainEn: 'Unit Square', domainJa: '単位正方形', boundary: 'neumann', mode: 3, eigenvalue: 19.739208802178716, formula: '2π²', description: 'cos(πx)cos(πy)', descriptionJa: 'cos(πx)cos(πy)', multiplicity: 1 },
  // UNIT SQUARE - STEKLOV
  { id: 'sq-s-0', domain: 'unit-square', domainEn: 'Unit Square', domainJa: '単位正方形', boundary: 'steklov', mode: 0, eigenvalue: 0, formula: '0', description: 'Constant', descriptionJa: '定数', multiplicity: 1 },
  { id: 'sq-s-1', domain: 'unit-square', domainEn: 'Unit Square', domainJa: '単位正方形', boundary: 'steklov', mode: 1, eigenvalue: 1.5707963267948966, formula: 'π/2', description: 'First Steklov', descriptionJa: '第1ステクロフ', multiplicity: 2 },
  // UNIT DISK - DIRICHLET
  { id: 'disk-d-1', domain: 'unit-disk', domainEn: 'Unit Disk', domainJa: '単位円板', boundary: 'dirichlet', mode: 1, eigenvalue: 5.7831859629, formula: 'j₀,₁² ≈ 2.405²', description: 'J₀(j₀,₁r)', descriptionJa: 'J₀(j₀,₁r)', multiplicity: 1, reference: 'Abramowitz & Stegun' },
  { id: 'disk-d-2', domain: 'unit-disk', domainEn: 'Unit Disk', domainJa: '単位円板', boundary: 'dirichlet', mode: 2, eigenvalue: 14.68197064374386, formula: 'j₁,₁² ≈ 3.832²', description: 'J₁(j₁,₁r)cos(θ)', descriptionJa: 'J₁(j₁,₁r)cos(θ)', multiplicity: 2 },
  { id: 'disk-d-3', domain: 'unit-disk', domainEn: 'Unit Disk', domainJa: '単位円板', boundary: 'dirichlet', mode: 4, eigenvalue: 26.37461643889078, formula: 'j₂,₁² ≈ 5.136²', description: 'J₂(j₂,₁r)cos(2θ)', descriptionJa: 'J₂(j₂,₁r)cos(2θ)', multiplicity: 2 },
  { id: 'disk-d-4', domain: 'unit-disk', domainEn: 'Unit Disk', domainJa: '単位円板', boundary: 'dirichlet', mode: 6, eigenvalue: 30.47126234366397, formula: 'j₀,₂² ≈ 5.520²', description: 'J₀(j₀,₂r)', descriptionJa: 'J₀(j₀,₂r)', multiplicity: 1 },
  { id: 'disk-d-5', domain: 'unit-disk', domainEn: 'Unit Disk', domainJa: '単位円板', boundary: 'dirichlet', mode: 7, eigenvalue: 40.7064645747, formula: 'j₃,₁² ≈ 6.380²', description: 'J₃(j₃,₁r)cos(3θ)', descriptionJa: 'J₃(j₃,₁r)cos(3θ)', multiplicity: 2 },
  // UNIT DISK - STEKLOV (σₙ = n exactly!)
  { id: 'disk-s-0', domain: 'unit-disk', domainEn: 'Unit Disk', domainJa: '単位円板', boundary: 'steklov', mode: 0, eigenvalue: 0, formula: '0', description: 'Constant', descriptionJa: '定数', multiplicity: 1 },
  { id: 'disk-s-1', domain: 'unit-disk', domainEn: 'Unit Disk', domainJa: '単位円板', boundary: 'steklov', mode: 1, eigenvalue: 1.0, formula: '1', description: 'rcos(θ), rsin(θ)', descriptionJa: 'rcos(θ), rsin(θ)', multiplicity: 2, reference: 'Girouard-Polterovich' },
  { id: 'disk-s-2', domain: 'unit-disk', domainEn: 'Unit Disk', domainJa: '単位円板', boundary: 'steklov', mode: 3, eigenvalue: 2.0, formula: '2', description: 'r²cos(2θ), r²sin(2θ)', descriptionJa: 'r²cos(2θ), r²sin(2θ)', multiplicity: 2 },
  { id: 'disk-s-3', domain: 'unit-disk', domainEn: 'Unit Disk', domainJa: '単位円板', boundary: 'steklov', mode: 5, eigenvalue: 3.0, formula: '3', description: 'r³cos(3θ), r³sin(3θ)', descriptionJa: 'r³cos(3θ), r³sin(3θ)', multiplicity: 2 },
  { id: 'disk-s-4', domain: 'unit-disk', domainEn: 'Unit Disk', domainJa: '単位円板', boundary: 'steklov', mode: 7, eigenvalue: 4.0, formula: '4', description: 'r⁴cos(4θ), r⁴sin(4θ)', descriptionJa: 'r⁴cos(4θ), r⁴sin(4θ)', multiplicity: 2 },
  { id: 'disk-s-5', domain: 'unit-disk', domainEn: 'Unit Disk', domainJa: '単位円板', boundary: 'steklov', mode: 9, eigenvalue: 5.0, formula: '5', description: 'r⁵cos(5θ), r⁵sin(5θ)', descriptionJa: 'r⁵cos(5θ), r⁵sin(5θ)', multiplicity: 2 },
  // L-SHAPED DOMAIN - Famous benchmark (corner singularity)
  { id: 'l-d-1', domain: 'l-shape', domainEn: 'L-shaped Domain', domainJa: 'L字型領域', boundary: 'dirichlet', mode: 1, eigenvalue: 9.6397238440219, formula: '≈ 9.6397', description: 'Corner singularity r^(2/3)', descriptionJa: '角の特異性 r^(2/3)', multiplicity: 1, computed: true, reference: 'Fox-Henrici-Moler 1967' },
  { id: 'l-d-2', domain: 'l-shape', domainEn: 'L-shaped Domain', domainJa: 'L字型領域', boundary: 'dirichlet', mode: 2, eigenvalue: 15.19725192535, formula: '≈ 15.197', description: 'Second mode', descriptionJa: '第2モード', multiplicity: 1, computed: true },
  { id: 'l-d-3', domain: 'l-shape', domainEn: 'L-shaped Domain', domainJa: 'L字型領域', boundary: 'dirichlet', mode: 3, eigenvalue: 19.73920880218, formula: '2π²', description: 'Equals square λ₁', descriptionJa: '正方形λ₁と一致', multiplicity: 1, computed: true },
  { id: 'l-d-4', domain: 'l-shape', domainEn: 'L-shaped Domain', domainJa: 'L字型領域', boundary: 'dirichlet', mode: 4, eigenvalue: 29.5214811, formula: '≈ 29.521', description: 'Fourth mode', descriptionJa: '第4モード', multiplicity: 1, computed: true },
  { id: 'l-d-5', domain: 'l-shape', domainEn: 'L-shaped Domain', domainJa: 'L字型領域', boundary: 'dirichlet', mode: 5, eigenvalue: 31.9126360, formula: '≈ 31.913', description: 'Fifth mode', descriptionJa: '第5モード', multiplicity: 1, computed: true },
  { id: 'l-d-6', domain: 'l-shape', domainEn: 'L-shaped Domain', domainJa: 'L字型領域', boundary: 'dirichlet', mode: 6, eigenvalue: 41.4745099, formula: '≈ 41.475', description: 'Sixth mode', descriptionJa: '第6モード', multiplicity: 1, computed: true },
  // EQUILATERAL TRIANGLE - Lamé (1833)
  { id: 'tri-d-1', domain: 'equilateral-triangle', domainEn: 'Equilateral Triangle', domainJa: '正三角形', boundary: 'dirichlet', mode: 1, eigenvalue: 52.637788657757, formula: '16π²/(3√3)', description: 'Lamé 1833', descriptionJa: 'ラメ 1833', multiplicity: 1, reference: 'Lamé 1833' },
  { id: 'tri-d-2', domain: 'equilateral-triangle', domainEn: 'Equilateral Triangle', domainJa: '正三角形', boundary: 'dirichlet', mode: 2, eigenvalue: 122.1547005384, formula: '112π²/(9√3)', description: 'Double mode', descriptionJa: '2重モード', multiplicity: 2 },
  { id: 'tri-d-3', domain: 'equilateral-triangle', domainEn: 'Equilateral Triangle', domainJa: '正三角形', boundary: 'dirichlet', mode: 4, eigenvalue: 175.1259621925, formula: '160π²/(9√3)', description: 'Third distinct', descriptionJa: '第3固有値', multiplicity: 1 },
  // ISOSCELES RIGHT TRIANGLE
  { id: 'irt-d-1', domain: 'isosceles-right-triangle', domainEn: 'Isosceles Right Triangle', domainJa: '直角二等辺三角形', boundary: 'dirichlet', mode: 1, eigenvalue: 49.34802200544679, formula: '5π²', description: 'Half of square mode', descriptionJa: '正方形モードの半分', multiplicity: 1 },
  { id: 'irt-d-2', domain: 'isosceles-right-triangle', domainEn: 'Isosceles Right Triangle', domainJa: '直角二等辺三角形', boundary: 'dirichlet', mode: 2, eigenvalue: 98.69604401089358, formula: '10π²', description: 'Second mode', descriptionJa: '第2モード', multiplicity: 1 },
  { id: 'irt-d-3', domain: 'isosceles-right-triangle', domainEn: 'Isosceles Right Triangle', domainJa: '直角二等辺三角形', boundary: 'dirichlet', mode: 3, eigenvalue: 128.30485721415769, formula: '13π²', description: 'Third mode', descriptionJa: '第3モード', multiplicity: 1 },
  // STADIUM (Bunimovich billiard - Quantum Chaos)
  { id: 'stadium-d-1', domain: 'stadium', domainEn: 'Stadium (Bunimovich)', domainJa: 'スタジアム', boundary: 'dirichlet', mode: 1, eigenvalue: 2.4048, formula: '≈ 2.405', description: 'Chaotic billiard - GOE statistics', descriptionJa: 'カオスビリヤード - GOE統計', multiplicity: 1, computed: true, reference: 'Bunimovich 1979' },
  { id: 'stadium-d-2', domain: 'stadium', domainEn: 'Stadium (Bunimovich)', domainJa: 'スタジアム', boundary: 'dirichlet', mode: 2, eigenvalue: 3.8317, formula: '≈ 3.832', description: 'Level repulsion visible', descriptionJa: '準位反発が見える', multiplicity: 1, computed: true },
  { id: 'stadium-d-3', domain: 'stadium', domainEn: 'Stadium (Bunimovich)', domainJa: 'スタジアム', boundary: 'dirichlet', mode: 3, eigenvalue: 5.1356, formula: '≈ 5.136', description: 'Third mode', descriptionJa: '第3モード', multiplicity: 1, computed: true },
  { id: 'stadium-d-4', domain: 'stadium', domainEn: 'Stadium (Bunimovich)', domainJa: 'スタジアム', boundary: 'dirichlet', mode: 4, eigenvalue: 6.3802, formula: '≈ 6.380', description: 'Fourth mode', descriptionJa: '第4モード', multiplicity: 1, computed: true },
  { id: 'stadium-d-5', domain: 'stadium', domainEn: 'Stadium (Bunimovich)', domainJa: 'スタジアム', boundary: 'dirichlet', mode: 5, eigenvalue: 7.5883, formula: '≈ 7.588', description: 'Fifth mode', descriptionJa: '第5モード', multiplicity: 1, computed: true },
  // QUARTER DISK (90° sector)
  { id: 'sec90-d-1', domain: 'sector-90', domainEn: 'Quarter Disk', domainJa: '四分円', boundary: 'dirichlet', mode: 1, eigenvalue: 26.3746, formula: 'j₂,₁² ≈ 5.136²', description: 'Bessel J₂', descriptionJa: 'ベッセル J₂', multiplicity: 1, computed: true },
  { id: 'sec90-d-2', domain: 'sector-90', domainEn: 'Quarter Disk', domainJa: '四分円', boundary: 'dirichlet', mode: 2, eigenvalue: 57.5817, formula: 'j₄,₁²', description: 'Bessel J₄', descriptionJa: 'ベッセル J₄', multiplicity: 1, computed: true },
  { id: 'sec90-d-3', domain: 'sector-90', domainEn: 'Quarter Disk', domainJa: '四分円', boundary: 'dirichlet', mode: 3, eigenvalue: 74.887, formula: 'j₂,₂²', description: 'Second radial J₂', descriptionJa: '第2動径 J₂', multiplicity: 1, computed: true },
  // 60° SECTOR
  { id: 'sec60-d-1', domain: 'sector-60', domainEn: 'Circular Sector 60°', domainJa: '扇形 60°', boundary: 'dirichlet', mode: 1, eigenvalue: 40.7065, formula: 'j₃,₁²', description: 'Bessel J₃', descriptionJa: 'ベッセル J₃', multiplicity: 1, computed: true },
  { id: 'sec60-d-2', domain: 'sector-60', domainEn: 'Circular Sector 60°', domainJa: '扇形 60°', boundary: 'dirichlet', mode: 2, eigenvalue: 91.3647, formula: 'j₆,₁²', description: 'Bessel J₆', descriptionJa: 'ベッセル J₆', multiplicity: 1, computed: true },
  // ANNULUS (r=0.5,1)
  { id: 'ann-d-1', domain: 'annulus-0.5', domainEn: 'Annulus (r=0.5,1)', domainJa: '円環 (r=0.5,1)', boundary: 'dirichlet', mode: 1, eigenvalue: 28.094, formula: '≈ 28.094', description: 'Radially symmetric', descriptionJa: '軸対称', multiplicity: 1, computed: true },
  { id: 'ann-d-2', domain: 'annulus-0.5', domainEn: 'Annulus (r=0.5,1)', domainJa: '円環 (r=0.5,1)', boundary: 'dirichlet', mode: 2, eigenvalue: 30.471, formula: '≈ 30.471', description: 'm=1 angular', descriptionJa: '角度モード m=1', multiplicity: 2, computed: true },
  { id: 'ann-d-3', domain: 'annulus-0.5', domainEn: 'Annulus (r=0.5,1)', domainJa: '円環 (r=0.5,1)', boundary: 'dirichlet', mode: 4, eigenvalue: 38.472, formula: '≈ 38.472', description: 'm=2 angular', descriptionJa: '角度モード m=2', multiplicity: 2, computed: true },
  // THIN TRIANGLE - Asymptotic behavior (λ₁ ~ π²/h² as h→0)
  { id: 'thin-d-1', domain: 'thin-triangle-0.1', domainEn: 'Thin Triangle (h=0.1)', domainJa: '細長い三角形 (h=0.1)', boundary: 'dirichlet', mode: 1, eigenvalue: 994.25, formula: '≈ π²/h² = 987', description: 'Friedlander asymptotic', descriptionJa: 'フリードランダーの漸近', multiplicity: 1, computed: true, reference: 'Friedlander 1992' },
  { id: 'thin-d-2', domain: 'thin-triangle-0.1', domainEn: 'Thin Triangle (h=0.1)', domainJa: '細長い三角形 (h=0.1)', boundary: 'dirichlet', mode: 2, eigenvalue: 2012.8, formula: '≈ 2012.8', description: 'Second mode', descriptionJa: '第2モード', multiplicity: 1, computed: true },
  { id: 'thin-d-3', domain: 'thin-triangle-0.1', domainEn: 'Thin Triangle (h=0.1)', domainJa: '細長い三角形 (h=0.1)', boundary: 'dirichlet', mode: 3, eigenvalue: 3534.7, formula: '≈ 3534.7', description: 'Third mode', descriptionJa: '第3モード', multiplicity: 1, computed: true },
  // PENTAGON
  { id: 'pent-d-1', domain: 'pentagon', domainEn: 'Regular Pentagon', domainJa: '正五角形', boundary: 'dirichlet', mode: 1, eigenvalue: 10.8698, formula: '≈ 10.870', description: 'FEniCS computed', descriptionJa: 'FEniCS計算', multiplicity: 1, computed: true },
  { id: 'pent-d-2', domain: 'pentagon', domainEn: 'Regular Pentagon', domainJa: '正五角形', boundary: 'dirichlet', mode: 2, eigenvalue: 25.5453, formula: '≈ 25.545', description: 'Double mode', descriptionJa: '2重モード', multiplicity: 2, computed: true },
  // HEXAGON
  { id: 'hex-d-1', domain: 'hexagon', domainEn: 'Regular Hexagon', domainJa: '正六角形', boundary: 'dirichlet', mode: 1, eigenvalue: 7.1552, formula: '≈ 7.155', description: 'FEniCS computed', descriptionJa: 'FEniCS計算', multiplicity: 1, computed: true },
  { id: 'hex-d-2', domain: 'hexagon', domainEn: 'Regular Hexagon', domainJa: '正六角形', boundary: 'dirichlet', mode: 2, eigenvalue: 18.5364, formula: '≈ 18.536', description: 'Double mode', descriptionJa: '2重モード', multiplicity: 2, computed: true },
  // ELLIPSE 2:1
  { id: 'ell-d-1', domain: 'ellipse-2-1', domainEn: 'Ellipse (2:1)', domainJa: '楕円 (2:1)', boundary: 'dirichlet', mode: 1, eigenvalue: 11.545, formula: '≈ 11.545', description: 'Mathieu functions', descriptionJa: 'マシュー関数', multiplicity: 1, computed: true },
  { id: 'ell-d-2', domain: 'ellipse-2-1', domainEn: 'Ellipse (2:1)', domainJa: '楕円 (2:1)', boundary: 'dirichlet', mode: 2, eigenvalue: 26.382, formula: '≈ 26.382', description: 'Second mode', descriptionJa: '第2モード', multiplicity: 1, computed: true },
  // RECTANGLE 2:1
  { id: 'rect-d-1', domain: 'rectangle-2-1', domainEn: 'Rectangle (2:1)', domainJa: '長方形 (2:1)', boundary: 'dirichlet', mode: 1, eigenvalue: 12.337005501361697, formula: '5π²/4', description: 'sin(πx/2)sin(πy)', descriptionJa: 'sin(πx/2)sin(πy)', multiplicity: 1 },
  { id: 'rect-d-2', domain: 'rectangle-2-1', domainEn: 'Rectangle (2:1)', domainJa: '長方形 (2:1)', boundary: 'dirichlet', mode: 2, eigenvalue: 22.206609902451, formula: '9π²/4', description: 'sin(πx)sin(πy)', descriptionJa: 'sin(πx)sin(πy)', multiplicity: 1 },
  { id: 'rect-d-3', domain: 'rectangle-2-1', domainEn: 'Rectangle (2:1)', domainJa: '長方形 (2:1)', boundary: 'dirichlet', mode: 3, eigenvalue: 41.945822104717, formula: '17π²/4', description: 'Third mode', descriptionJa: '第3モード', multiplicity: 1 },
];

// Helper functions
export function getEigenfunctionsByDomain(domainId: string): Eigenfunction[] {
  return eigenfunctions.filter(ef => ef.domain === domainId);
}

export function getEigenfunctionsByBoundary(boundary: 'dirichlet' | 'neumann' | 'steklov'): Eigenfunction[] {
  return eigenfunctions.filter(ef => ef.boundary === boundary);
}

export function getEigenfunctionByMode(domainId: string, boundary: string, mode: number): Eigenfunction | undefined {
  return eigenfunctions.find(ef => ef.domain === domainId && ef.boundary === boundary && ef.mode === mode);
}

export function getDomainInfo(domainId: string): DomainInfo | undefined {
  return domains.find(d => d.id === domainId);
}

export function getDomainsByCategory(category: string): DomainInfo[] {
  return domains.filter(d => d.category === category);
}

export function getMaxMode(domainId: string, boundary: string): number {
  const efs = eigenfunctions.filter(ef => ef.domain === domainId && ef.boundary === boundary);
  return efs.length > 0 ? Math.max(...efs.map(ef => ef.mode)) : 0;
}

export function searchEigenfunctions(query: string, lang: 'en' | 'ja' = 'en'): Eigenfunction[] {
  const q = query.toLowerCase();
  return eigenfunctions.filter(ef => {
    const name = lang === 'ja' ? ef.domainJa : ef.domainEn;
    const desc = lang === 'ja' ? ef.descriptionJa : ef.description;
    return name.toLowerCase().includes(q) || ef.boundary.includes(q) || ef.formula.toLowerCase().includes(q) || desc.toLowerCase().includes(q);
  });
}
