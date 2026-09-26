-- Cast-shadow pass: the sun's own render of the world.
--
-- The scene shader already asks, per fragment, whether anything stood
-- between it and the sun (Voxel3D's sunlight()); this module is what
-- FILLS the map it asks against.  An orthographic camera down the sun
-- line stores, per texel, how far the light travelled before it hit
-- something; the main pass transforms each fragment into that same space
-- and compares.  What the sun cannot see is in shadow, whatever surface
-- it happens to be -- so a shadow climbs a wall, drapes over a roof and
-- needs no per-case code: every caster is simply whatever this pass draws
-- (the terrain and cap meshes -- FrTerrain.cast).
--
-- Depth is stored in an ORDINARY colour canvas, packed into two 8-bit
-- channels (~16 bits over the frustum).  A readable depth texture would be
-- tidier, but depth sampling is the least portable corner of the graphics
-- API: everything here is pcall-guarded and `available()` reports the
-- result, and the scene simply stays unshadowed (sunDark = 0) when it says
-- no.  The scene shader unpacks exactly this pack (r + g/255).
--
-- Origin: DramaticShapeVoxelMod (lib/ShadowMap.lua), reduced to the
-- surface this scene reads and ported to this mod's module set (Mat4,
-- VoxelState; canvas creation in line).

-- the mod namespace (see main.lua): V.require loads a sibling module
local V = ...

local Mat4 = V.require("Mat4")
local Voxel = V.require("VoxelState")

local ShadowMap = {}

-- The sun, as the shear a shadow takes: a point `y` world-pixels above
-- the ground drops its shadow (KX*y, KZ*y) away from the point under it.
-- Both negative hangs the sun in the SOUTHEAST, so every shadow falls
-- northwest -- up and to the left on screen (north at the top, east at
-- the right at every tilt), and the baked FACE_SHADE in Voxel3D already
-- treats east/south as the lit flanks to match.
--
-- Magnitude: hypot(KX, KZ) = 1.01 is a sun about 45 degrees up, so a 16px
-- wall throws a shadow about as long as it is tall.  Ratio: west of
-- northwest on purpose, so a shadow clears what casts it on screen.
ShadowMap.KX = -0.85      -- west drift per pixel of height
ShadowMap.KZ = -0.55      -- north drift per pixel of height

-- Shadow map edge, in texels, picked per frame from this ladder (the
-- light frustum is sized to the world VIEW, which swings with zoom and
-- window size).  TARGET is world pixels per texel worth paying for: a
-- third of a pixel puts the shadow edge inside the pixel grid this whole
-- mode exists to keep crisp.
ShadowMap.SIZES = { 1024, 1536, 2048 }
ShadowMap.TARGET = 0.45
ShadowMap.res = 1024      -- the rung in use; read by the main pass filter

-- The tallest geometry the pass covers: gabled buildings and border
-- forest run well under this, and the margin buys casts from off-screen.
ShadowMap.HEIGHT = 160

-- Depth slack at the comparison, in world pixels.  Too little and a lit
-- surface shadows itself in moire acne; too much and a shadow detaches
-- from the foot of what casts it.  Cannot be ONE number, because what the
-- comparison forgives scales with the texel: BIAS covers what does not
-- (quantisation, the two passes reaching a point by different matrices),
-- SLOPE covers the depth ramp a lit surface reads across one texel.
ShadowMap.BIAS = 0.5
ShadowMap.SLOPE = 3.1
ShadowMap.slack = ShadowMap.BIAS

local SHADER = [[
  varying float vDepth;
#ifdef VERTEX
  uniform mat4 lightVP;
  uniform mat4 model;
  vec4 position(mat4 transform_projection, vec4 vertex_position) {
    vec4 c = lightVP * (model * vertex_position);
    // the projection is orthographic (w is 1) and fit() maps clip z onto
    // [0,1] directly (see Z01 there), so clip z IS the stored depth
    vDepth = c.z;
    return c;
  }
#endif
#ifdef PIXEL
  uniform float sprite;   // 1 while the CAST is being drawn
  vec4 effect(vec4 color, Image tex, vec2 tc, vec2 sc) {
    // the same alpha discard the main pass uses
    if (Texel(tex, tc).a < 0.5) discard;
    // pack into two channels: high byte in red, low in green.  Blue says
    // WHAT cast this, and lets a surface decline one kind of caster later.
    float d = clamp(vDepth, 0.0, 1.0) * 255.0;
    return vec4(floor(d) / 255.0, fract(d), sprite, 1.0);
  }
#endif
]]

ShadowMap._source = function() return SHADER end   -- named for the suite

local shader = nil            -- nil = untried, false = unavailable
local canvas = nil            -- nil = untried, false = unavailable
local canvasRes = 0
local blank = nil             -- 1x1 stand-in so the sampler is never unbound
local drawing = false
local ready = false
local lastSig = nil
local prevBlend, prevAlphaMode = nil, nil

local IDENTITY = Mat4.identity()

-- world -> [0,1] cube, applied on top of the clip matrix: the main pass
-- samples the map with the xy and compares against the z.  The V ROW's
-- sign is the RUNTIME's to answer: this pass bypasses transform_projection
-- (the one seam where LOVE reconciles clip conventions) and LOVE 12
-- flipped clip +y, so a draw this pass stores lands with clip +y at
-- texture v 1 under LOVE 11 and v 0 under LOVE 12 -- probeVSign()
-- measures which world this is once, with a real draw, instead of
-- trusting a version list.  The z row is identity: fit() already maps
-- clip z onto [0,1] (see Z01).
local function toUnit(vSign)
  return { 0.5, 0, 0, 0.5,
           0, 0.5 * vSign, 0, 0.5,
           0, 0, 1, 0,
           0, 0, 0, 1 }
end

-- Clip z from GL's [-1,1] onto [0,1], multiplied onto the projection:
-- Mat4.ortho emits the legacy GL range, and the reader compares packed
-- [0,1] depth, so without this the near half of the light frustum comes
-- back MISSING (half-empty maps cut along the z = 0 plane).
local Z01 = { 1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 0.5, 0.5,
              0, 0, 0, 1 }

-- +1 = LOVE 11 storage orientation, -1 = LOVE 12's; nil until probed
local vSign = nil

-- world -> light clip space, for the pass that FILLS the map
ShadowMap.clipVP = IDENTITY
-- world -> the unit cube, for the pass that READS it
ShadowMap.uvVP = IDENTITY
-- ShadowMap.BIAS expressed in the [0,1] depth the map stores
ShadowMap.bias = 0

local function getShader()
  if shader == nil then
    local ok, sh = pcall(love.graphics.newShader, SHADER)
    shader = (ok and sh) or false
  end
  return shader or nil
end

-- The map canvas at edge `res`, rebuilt when the rung changes.  `false`
-- is sticky: a driver that could not make one is not asked every frame.
local function getCanvas(res)
  if canvas == false then return nil end
  if canvas and canvasRes == res then return canvas end
  local ok, c = pcall(love.graphics.newCanvas, res, res)
  if not ok or not c then
    canvas = false
    return nil
  end
  -- nearest: the 2x2 filter in the main pass wants raw texels, and a
  -- linearly blended PACKED depth is not a depth at all
  pcall(c.setFilter, c, "nearest", "nearest")
  pcall(c.setWrap, c, "clamp", "clamp")
  if canvas and canvas.release then pcall(canvas.release, canvas) end
  canvas, canvasRes = c, res
  ready = false
  return canvas
end

-- A 1x1 opaque white image: the main pass always declares the shadow
-- sampler, so something must be bound even on frames with no map --
-- unpacked it reads as depth 1 + 1/255, past the far plane, i.e.
-- "nothing occludes anything".
local function getBlank()
  if blank == nil then
    local ok, img = pcall(function()
      local data = love.image.newImageData(1, 1)
      data:setPixel(0, 0, 1, 1, 1, 1)
      return love.graphics.newImage(data)
    end)
    blank = (ok and img) or false
  end
  return blank or nil
end

-- Which way a bypass-projection draw lands in a canvas on THIS runtime,
-- measured with a real draw rather than assumed: write half of clip space
-- through the pass's own shader and ask which rows of the readback it
-- wrote.  Any failure answers +1 (the convention LOVE shipped on until
-- 12), so a driver that refuses the readback keeps its behaviour.
local function probeVSign()
  local done, sign = pcall(function()
    local sh = getShader()
    local tex = getBlank()
    if not (sh and tex) then return 1 end
    -- dpiscale 1: the readback below indexes rows of the IMAGE
    local c = love.graphics.newCanvas(4, 4, { dpiscale = 1 })
    local mesh = love.graphics.newMesh(
      { { -1, 0, 0, 0 }, { 1, 0, 1, 0 }, { 1, 1, 1, 1 }, { -1, 1, 0, 1 } },
      "fan", "static")
    mesh:setTexture(tex)
    love.graphics.setCanvas(c)
    love.graphics.clear(1, 1, 0, 1)
    love.graphics.setShader(sh)
    love.graphics.setColor(1, 1, 1, 1)
    pcall(sh.send, sh, "lightVP", "row",
          Mat4.mul(Z01, Mat4.scale(1, -1, 1)))
    pcall(sh.send, sh, "model", "row", IDENTITY)
    pcall(sh.send, sh, "sprite", 0)
    love.graphics.draw(mesh)
    love.graphics.setShader()
    love.graphics.setCanvas()
    local data = c:newImageData()
    local w, h = data:getDimensions()
    -- written texels carry packed depth 0.5 (red ~0.5); the clear is red 1
    local topR = data:getPixel(1, 0)
    local botR = data:getPixel(1, h - 1)
    if c.release then pcall(c.release, c) end
    if topR < 0.9 and botR > 0.9 then return 1 end
    if botR < 0.9 and topR > 0.9 then return -1 end
    return 1
  end)
  return (done and sign) or 1
end

-- Whether the sun pass can run at all: false headless, without shaders,
-- or where the canvas cannot be made -- the scene then keeps its baked
-- face shading with sunDark = 0, which needs nothing.
function ShadowMap.available()
  if not (love.graphics and love.graphics.newCanvas
          and love.graphics.setDepthMode) then
    return false
  end
  if not getShader() then return false end
  if canvas ~= nil then return canvas ~= false end
  return getCanvas(ShadowMap.SIZES[1]) ~= nil
end

-- The map to sample, or the blank stand-in.  Never nil once the scene
-- has a shader at all, because an unbound sampler is a driver-dependent
-- crash rather than a driver-dependent fallback.
function ShadowMap.texture()
  if ready and canvas then return canvas end
  return getBlank()
end

-- True while the map holds a frame the main pass can read.
function ShadowMap.active()
  return ready and canvas ~= nil and canvas ~= false
end

-- Drop the last completed map without releasing its storage: a frame
-- that must not see shadows (a menu over an empty stage) calls this and
-- the next real scene recasts.
function ShadowMap.discard()
  ready, lastSig = false, nil
end

-- The direction the light TRAVELS, normalized: the shear is the shadow
-- a unit of height throws, so the ray from a point to its shadow is
-- (KX, -1, KZ).
local function sunDir()
  local x, y, z = ShadowMap.KX, -1, ShadowMap.KZ
  local l = math.sqrt(x * x + y * y + z * z)
  return { x / l, y / l, z / l }
end

ShadowMap.sunDir = sunDir

-- How far NORTH of the view centre the camera can still see ground, in
-- world pixels: the top edge of the view frustum dropped onto the ground
-- plane, capped (past about 64 degrees the ray clears the horizon and the
-- honest answer is "forever").  Past the cap the far field's shadows ease
-- out at the rim instead of the frustum ending on a hard line.
ShadowMap.FAR_CAP = 2.5     -- multiples of the view height

local function groundReach(vh)
  local a = Voxel.angle or 0
  local cap = ShadowMap.FAR_CAP * vh
  local half = math.atan(1 / (2 * Voxel.FOCAL))
  local below = (math.pi / 2 - a) - half     -- top ray, below horizontal
  if below <= 0.02 then return cap end
  local dist = Voxel.FOCAL * vh
  local horizon = dist * math.cos(a) / math.tan(below)
  return math.max(vh / 2, math.min(cap, horizon - dist * math.sin(a)))
end

-- Fit the light frustum to the ground the camera can see, plus the margin
-- the casters for it stand in.  Both ASYMMETRIC: the camera sits south of
-- its focus looking north (ground runs far north, barely south), and the
-- sun sits southeast (the casters stand south and east of what they
-- shadow) -- so the margin is only ever needed on two of the four sides.
--
-- The box is snapped to whole texels: without that, a frustum that slides
-- with the camera reprojects every shadow edge a fraction of a texel
-- every frame and the world's shadows crawl while you walk.
local function fit(cx, cy, vw, vh)
  local f = sunDir()
  local view = Mat4.lookAt({ 0, 0, 0 }, f, { 0, 0, -1 })

  local reach = ShadowMap.HEIGHT
                * math.max(math.abs(ShadowMap.KX), math.abs(ShadowMap.KZ)) + 24
  local north = groundReach(vh)
  local spread = north * 0.5
  local xs = { cx - vw / 2 - spread, cx + vw / 2 + spread + reach }
  local ys = { -32, ShadowMap.HEIGHT }         -- -32 covers recessed water
  local zs = { cy - north, cy + vh / 2 + reach }

  local l, r, b, t, zn, zf
  for _, x in ipairs(xs) do
    for _, y in ipairs(ys) do
      for _, z in ipairs(zs) do
        local px = view[1] * x + view[2] * y + view[3] * z + view[4]
        local py = view[5] * x + view[6] * y + view[7] * z + view[8]
        local pz = view[9] * x + view[10] * y + view[11] * z + view[12]
        l = l and math.min(l, px) or px
        r = r and math.max(r, px) or px
        b = b and math.min(b, py) or py
        t = t and math.max(t, py) or py
        zn = zn and math.min(zn, pz) or pz
        zf = zf and math.max(zf, pz) or pz
      end
    end
  end

  local w, h = r - l, t - b

  -- smallest rung that resolves TARGET world pixels per texel, else the
  -- largest there is
  local res = ShadowMap.SIZES[#ShadowMap.SIZES]
  for _, size in ipairs(ShadowMap.SIZES) do
    if math.max(w, h) / size <= ShadowMap.TARGET then
      res = size
      break
    end
  end
  ShadowMap.res = res

  -- the box's SIZE is fixed (sun and view size are), so snapping its
  -- corner to a texel multiple moves it in whole texels only
  local tx, ty = w / res, h / res
  l = math.floor(l / tx) * tx
  b = math.floor(b / ty) * ty
  r, t = l + w, b + h

  -- view-space z runs NEGATIVE into the scene; ortho() wants distances,
  -- and the slack keeps geometry taller than HEIGHT from being clipped
  -- clean out of the pass instead of merely casting truncated
  local near, far = -zf - 64, -zn + 64
  local proj = Mat4.ortho(l, r, b, t, near, far)
  -- flip clip-space Y: we bypass LOVE's transform_projection and canvas
  -- coordinates run Y DOWN, so without this the map is stored upside
  -- down relative to the uv the main pass reads it with
  proj = Mat4.mul(Mat4.scale(1, -1, 1), proj)
  -- and clip z onto [0,1] -- see Z01 for why this is load-bearing
  proj = Mat4.mul(Z01, proj)

  ShadowMap.clipVP = Mat4.mul(proj, view)
  ShadowMap.uvVP = Mat4.mul(toUnit(vSign or 1), ShadowMap.clipVP)
  -- the slack the comparison needs, against the coarser of the two texel
  -- axes (the box is asymmetric, and one number has to cover both)
  ShadowMap.slack = ShadowMap.BIAS
                    + ShadowMap.SLOPE * math.max(w, h) / res
  -- the stored depth spans the frustum, so a world-pixel bias is that
  -- fraction of it
  ShadowMap.bias = ShadowMap.slack / math.max(1, far - near)
end

-- Whether the map has to be redrawn for `sig` -- a caller-built stamp of
-- everything the pass depends on (camera, terrain mesh, poses).  A frame
-- that changes none of it reuses the map it already has.
function ShadowMap.stale(sig)
  return not ready or sig ~= lastSig
end

-- Begin the sun pass.  Returns false when it could not start, in which
-- case the caller must not draw into it or call finish.
function ShadowMap.begin(cx, cy, vw, vh)
  local sh = getShader()
  if not sh then return false end
  if vSign == nil then
    vSign = probeVSign()
    ShadowMap.vSign = vSign
  end
  -- fit first: it decides which resolution rung this view wants
  fit(cx, cy, vw, vh)
  local c = getCanvas(ShadowMap.res)
  if not c then return false end
  local ok = pcall(love.graphics.setCanvas, { c, depth = true })
  if not ok then
    pcall(love.graphics.setCanvas)
    canvas = false
    return false
  end
  prevBlend, prevAlphaMode = love.graphics.getBlendMode()
  -- white clears to depth 1 + 1/255, past the far plane: a texel nothing
  -- was drawn into can never shadow anything
  love.graphics.clear(1, 1, 0, 1, true, true)
  love.graphics.setDepthMode("lequal", true)
  love.graphics.setMeshCullMode("none")
  -- replace, not alpha blend: these are packed numbers, not colours
  love.graphics.setBlendMode("replace", "premultiplied")
  love.graphics.setShader(sh)
  love.graphics.setColor(1, 1, 1, 1)
  pcall(sh.send, sh, "lightVP", "row", ShadowMap.clipVP)
  pcall(sh.send, sh, "sprite", 0)
  drawing = true
  ready = false
  return true
end

-- Which kind of caster the next draws are (blue channel).  Kept for the
-- day characters cast too: a surface will be able to decline them.
function ShadowMap.sprites(on)
  if not drawing then return end
  local sh = getShader()
  if sh then pcall(sh.send, sh, "sprite", on and 1 or 0) end
end

-- Draw one caster.  Same signature as Voxel3D.draw minus the camera-ward
-- pull, which is a trick for the VIEW's depth buffer and would drag a
-- shadow off whatever throws it.
function ShadowMap.draw(mesh, texture, model)
  if not (drawing and mesh) then return end
  local sh = getShader()
  if texture then mesh:setTexture(texture) end
  pcall(sh.send, sh, "model", "row", model or IDENTITY)
  love.graphics.draw(mesh)
end

-- Close the pass and stamp it with the signature it was drawn for.
function ShadowMap.finish(sig)
  if not drawing then return end
  drawing = false
  love.graphics.setShader()
  love.graphics.setDepthMode()
  love.graphics.setCanvas()
  love.graphics.setBlendMode(prevBlend or "alpha", prevAlphaMode)
  love.graphics.setColor(1, 1, 1, 1)
  lastSig = sig
  ready = true
end

-- Drop the GPU objects (window resize, hot reload).
function ShadowMap.invalidate()
  if canvas and canvas.release then pcall(canvas.release, canvas) end
  canvas, canvasRes, blank = nil, 0, nil
  drawing, ready, lastSig = false, false, nil
end

return ShadowMap
