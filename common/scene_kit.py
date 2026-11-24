"""USD 기본 도형으로 공장 설비를 만드는 도구 모음 (외부 에셋 없이 동작 → 폐쇄망에서도 사용 가능)."""
from pathlib import Path

from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

HALL_USD = Path(__file__).resolve().parents[1] / "assets" / "sim" / "warehouse" / "warehouse.usd"


class Node:
    """움직이는 프림 하나. 위치·회전·크기·색·표시 여부를 빠르게 바꾸기 위한 손잡이."""

    def __init__(self, prim, t, r, s):
        self.prim, self._t, self._r, self._s = prim, t, r, s
        self._img = UsdGeom.Imageable(prim)
        self._vis = True
        self._color = None

    def pos(self, x, y, z):
        self._t.Set(Gf.Vec3d(float(x), float(y), float(z)))

    def rot(self, rx, ry, rz):
        self._r.Set(Gf.Vec3f(float(rx), float(ry), float(rz)))

    def scale(self, sx, sy, sz):
        self._s.Set(Gf.Vec3f(float(sx), float(sy), float(sz)))

    def show(self, on=True):
        if on != self._vis:
            self._vis = on
            # MakeVisible() 은 부모가 숨겨져 있으면 형제 프림을 숨겨 버리므로 속성을 직접 씀
            self._img.GetVisibilityAttr().Set("inherited" if on else "invisible")

    def color(self, rgb):
        if rgb != self._color:
            self._color = rgb
            UsdGeom.Gprim(self.prim).GetDisplayColorAttr().Set([Gf.Vec3f(*rgb)])


class Kit:
    def __init__(self, stage):
        self.stage = stage
        self._mats = {}
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Scope.Define(stage, "/World/Looks")

    # ---- 재질 ---------------------------------------------------------------
    def mat(self, color, rough=0.55, metal=0.0, opacity=1.0, emissive=None, name=None):
        key = name or (tuple(color), rough, metal, opacity, tuple(emissive) if emissive else None)
        if key in self._mats:
            return self._mats[key]
        path = f"/World/Looks/{name or 'M%03d' % len(self._mats)}"
        m = UsdShade.Material.Define(self.stage, path)
        sh = UsdShade.Shader.Define(self.stage, path + "/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        if emissive:
            sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        self._mats[key] = m
        m.shader = sh
        return m

    # ---- 도형 ---------------------------------------------------------------
    def _finish(self, geom, pos, rot, scale, color, mat):
        xf = UsdGeom.Xformable(geom)
        t = xf.AddTranslateOp()
        r = xf.AddRotateXYZOp()
        s = xf.AddScaleOp()
        t.Set(Gf.Vec3d(*[float(v) for v in pos]))
        r.Set(Gf.Vec3f(*[float(v) for v in rot]))
        s.Set(Gf.Vec3f(*[float(v) for v in scale]))
        node = Node(geom.GetPrim(), t, r, s)
        if mat is not None:
            UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(mat)
        elif color is not None:
            node.color(tuple(color))
        return node

    def group(self, path, pos=(0, 0, 0), rot=(0, 0, 0)):
        return self._finish(UsdGeom.Xform.Define(self.stage, path), pos, rot, (1, 1, 1), None, None)

    def box(self, path, size, pos=(0, 0, 0), color=None, mat=None, rot=(0, 0, 0)):
        g = UsdGeom.Cube.Define(self.stage, path)
        g.CreateSizeAttr(1.0)
        return self._finish(g, pos, rot, size, color, mat)

    def cyl(self, path, radius, height, pos=(0, 0, 0), axis="Z", color=None, mat=None, rot=(0, 0, 0)):
        g = UsdGeom.Cylinder.Define(self.stage, path)
        g.CreateRadiusAttr(1.0)
        g.CreateHeightAttr(1.0)
        g.CreateAxisAttr(axis)
        sc = {"X": (height, radius, radius), "Y": (radius, height, radius), "Z": (radius, radius, height)}[axis]
        return self._finish(g, pos, rot, sc, color, mat)

    def cone(self, path, radius, height, pos=(0, 0, 0), axis="Z", color=None, mat=None, rot=(0, 0, 0)):
        g = UsdGeom.Cone.Define(self.stage, path)
        g.CreateRadiusAttr(1.0)
        g.CreateHeightAttr(1.0)
        g.CreateAxisAttr(axis)
        sc = {"X": (height, radius, radius), "Y": (radius, height, radius), "Z": (radius, radius, height)}[axis]
        return self._finish(g, pos, rot, sc, color, mat)

    def sphere(self, path, radius, pos=(0, 0, 0), color=None, mat=None, scale=(1, 1, 1)):
        g = UsdGeom.Sphere.Define(self.stage, path)
        g.CreateRadiusAttr(1.0)
        return self._finish(g, pos, (0, 0, 0), tuple(radius * s for s in scale), color, mat)

    def line(self, path, points, width=0.03, color=(0.1, 0.5, 0.9)):
        c = UsdGeom.BasisCurves.Define(self.stage, path)
        c.CreateTypeAttr("linear")
        c.CreatePointsAttr([Gf.Vec3f(*[float(v) for v in p]) for p in points])
        c.CreateCurveVertexCountsAttr([len(points)])
        c.CreateWidthsAttr([width] * len(points))
        c.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
        c.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        return c

    def lights(self, dome=650.0, sun=1800.0):
        d = UsdLux.DomeLight.Define(self.stage, "/World/Lights/Dome")
        d.CreateIntensityAttr(float(dome))
        d.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))
        d.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)   # 배경은 검게
        if sun:
            s = UsdLux.DistantLight.Define(self.stage, "/World/Lights/Sun")
            s.CreateIntensityAttr(float(sun))
            s.CreateAngleAttr(2.0)
            UsdGeom.Xformable(s).AddRotateXYZOp().Set(Gf.Vec3f(48.0, 0.0, 24.0))

    def hall(self, pos=(0, 0, 0), yaw=0.0, scale=1.0, hide=None):
        """공장 건물(창고 에셋)을 배경으로 놓음. 에셋이 없으면 False (./scripts/factory assets 로 받음).

        hide(이름, 원래 좌표의 최소점, 최대점) 이 True 인 부재는 숨김 (카메라 쪽 벽을 떼어 낼 때 씀)."""
        if not HALL_USD.exists():
            return False
        prim = UsdGeom.Xform.Define(self.stage, "/World/Hall").GetPrim()
        prim.GetReferences().AddReference(str(HALL_USD))
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
        xf.AddRotateZOp().Set(float(yaw))
        xf.AddScaleOp().Set(Gf.Vec3f(float(scale)))
        from pxr import Usd
        self.hall_lights = [q for q in Usd.PrimRange(prim) if q.HasAPI(UsdLux.LightAPI)]
        if hide:
            bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
            inv = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
            for c in prim.GetChildren():
                r = bb.ComputeWorldBound(c).ComputeAlignedRange()
                if r.IsEmpty():
                    continue
                a, b = inv.Transform(r.GetMin()), inv.Transform(r.GetMax())
                lo = [min(a[i], b[i]) for i in range(3)]; hi = [max(a[i], b[i]) for i in range(3)]
                if hide(c.GetName(), lo, hi):
                    UsdGeom.Imageable(c).GetVisibilityAttr().Set("invisible")
        return True

    def area_light(self, path, pos, size=(2.0, 2.0), intensity=6000.0, color=(1.0, 0.97, 0.92)):
        """아래를 비추는 천장 조명."""
        lt = UsdLux.RectLight.Define(self.stage, path)
        lt.CreateWidthAttr(float(size[0])); lt.CreateHeightAttr(float(size[1]))
        lt.CreateIntensityAttr(float(intensity)); lt.CreateColorAttr(Gf.Vec3f(*color))
        UsdGeom.Xformable(lt).AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
        return lt


class Path3:
    """꺾은선 위를 일정 속도로 지나가는 위치 계산 (데이터 패킷 이동용)."""

    def __init__(self, points):
        self.pts = [Gf.Vec3d(*p) for p in points]
        self.seg = [(self.pts[i + 1] - self.pts[i]).GetLength() for i in range(len(self.pts) - 1)]
        self.length = sum(self.seg)

    def at(self, d):
        d = max(0.0, min(d, self.length))
        for i, L in enumerate(self.seg):
            if d <= L or i == len(self.seg) - 1:
                u = d / L if L > 0 else 0.0
                p = self.pts[i] + (self.pts[i + 1] - self.pts[i]) * u
                return p[0], p[1], p[2]
            d -= L
