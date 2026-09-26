-- Terrain geometry for the FireRed diorama.
--
-- The 2D path blits the current map (and its connected neighbours) as
-- metatile layers; here the same cells become extruded columns.  For each
-- cell the mesh carries the metatile's own atlas quad, so the ground
-- looks exactly like the flat game and the sides are the same tile
-- stretched down the height of the step -- which is what makes a house
-- roof, a tree canopy or the border ring read as a solid block instead of
-- a sticker.
--
-- Heights are DERIVED, not authored: there is no FireRed voxel-height
-- table, so the collision byte is the only signal the game itself has for
-- "you cannot stand here".  A cell the player can walk on stays at the
-- ground plane, a cell they cannot is raised one tile, and water is sunk.
-- That is enough for Pallet Town: buildings, fences and the border
-- extrude, while roads, grass and doorways stay flat.
--
-- Nothing here mutates the map: the collision array and the native
-- tileset atlas are read exactly as the 2D path reads them.

local V = ...
local Voxel3D = V.require("Voxel3D")

local FrTerrain = {}

local CELL = 16
-- Cells of apron meshed around the root map: enough to reach past the
-- connected neighbours the view can see, and to hide the void behind a
-- ring of border tiles (which the collision byte already reports as
-- solid, so it extrudes as a wall).
local APRON = 18
local MAX_CELLS = 9000
local WATER_DROP = 4
local SKIRT = -32

local heights = {}     -- flat row-major height of the BUILT rectangle
local scratch = {}      -- heights of the build under construction
local rect = nil        -- { x0, y0, W } describing `heights`
local signature = ""
local meshes = nil     -- pair -> { mesh = ..., image = ... }
local dirty = true
local warned = false

local function say(msg)
  pcall(print, "[fr_voxel] " .. msg)
end

-- The rebuild path can fail for a reason that is invisible from outside
-- (a tileset with no atlas, a driver that refuses the vertex format), and
-- a failure that repeats every frame repeats the log line too -- so these
-- go through the once-only `warned` latch like the two below.
local function warnOnce(msg)
  if warned then return end
  warned = true
  say(msg)
end

-- UV axis per face direction: which corner component drives the atlas U
-- and which drives V.  Top faces map V onto +Z so the tile's own top edge
-- faces north, exactly as the flat blit puts it.
local FACE_U = { [1] = 3, [2] = 3, [3] = 1, [4] = 1, [5] = 1, [6] = 1 }
local FACE_V = { [1] = 2, [2] = 2, [3] = 3, [4] = 3, [5] = 2, [6] = 2 }

local function engineRequire(name)
  local ok, m = pcall(require, name)
  if ok then return m end
  return nil
end

--- Which map owns a world cell: its definition, layout and the cell in
-- that layout's own space.  Falls back to the root layout, whose
-- midAt/collAt wrap onto the border ring -- the same fill the pret
-- connection code uses, so cells beyond every connected map still get a
-- tile and a collision byte rather than a hole.
local function ownerAt(rootDef, rootL, world, cx, cy)
  if cx >= 0 and cy >= 0 and cx < rootL.width and cy < rootL.height then
    return rootDef, rootL, cx, cy
  end
  for i = 1, #world do
    local e = world[i]
    local L = e.def and e.def.midLayout
    if L then
      local lx, ly = cx - e.ox, cy - e.oy
      if lx >= 0 and ly >= 0 and lx < L.width and ly < L.height then
        return e.def, L, lx, ly
      end
    end
  end
  return rootDef, rootL, cx, cy
end

local function passable(coll)
  local P = engineRequire("src.world.gen2.Permissions")
  if P and P.isWalkable then
    local ok, res = pcall(P.isWalkable, coll)
    if ok then return res and true or false end
  end
  return coll ~= 0x07 and coll ~= 0xff and coll ~= 0x29
end

local function isWater(def, lx, ly, coll)
  local okC, Collision = pcall(require, "src.core.game3.collision")
  if okC and Collision and Collision.isWaterOn then
    local ok, res = pcall(Collision.isWaterOn, def, lx, ly, coll)
    if ok then return res and true or false end
  end
  local P = engineRequire("src.world.gen2.Permissions")
  if P and P.isWater then
    local ok, res = pcall(P.isWater, coll)
    if ok then return res and true or false end
  end
  return coll == 0x29
end

-- NativeTileset.slotFor, but pcall'd: a mod must not be able to take the
-- frame down by asking a tileset it has no business asking.
local function NativeSlot(ts, mid)
  local ok, NativeTileset = pcall(require, "src.core.game3.tileset_native")
  if ok and NativeTileset and NativeTileset.slotFor then
    local okS, slot = pcall(NativeTileset.slotFor, ts, mid)
    if okS and type(slot) == "number" then return slot end
  end
  if type(ts.midToSlot) == "table" then return ts.midToSlot[mid] end
  return nil
end

--- The atlas image a tileset pair draws from.  nil means this map cannot
-- be meshed (no native tileset loaded for it).
local function atlasFor(pair)
  local okT, NativeTileset = pcall(require, "src.core.game3.tileset_native")
  if not (okT and NativeTileset and type(pair) == "string") then return nil end
  local ts = NativeTileset.get(pair)
  if not (ts and ts.image and ts.cols) then return nil end
  return ts
end

--- The atlas rectangle one metatile draws from, plus its slot number (the
-- caller needs the slot to address the same block in the cap colour image).
local function uvFor(ts, mid)
  if not ts then return nil end
  local slot = NativeSlot(ts, mid)
  if type(slot) ~= "number" or slot < 0 then return nil end
  local sx = (slot % ts.cols) * CELL
  local sy = math.floor(slot / ts.cols) * CELL
  local iw, ih = ts.image:getDimensions()
  if not (iw and ih and iw > 0 and ih > 0) then return nil end
  return { u0 = sx / iw, v0 = sy / ih,
           u1 = (sx + CELL) / iw, v1 = (sy + CELL) / ih }, slot
end

-- ------------------------------------------- solid caps for raised tops --
--
-- The top face of an extruded cell samples its metatile, which is right
-- for the ground and wrong for a building: the facade carries WINDOWS, so
-- the "roof" of the diorama reads as a wall seen from above and the real
-- roof tiles (where the art has them) never show as a cap.  Raised cells
-- therefore take their top face from a generated COMPOSITE image: the mid
-- tile with the over tile laid over its opaque pixels, pixel for pixel --
-- exactly what the flat field blits, because a top face IS a top-down
-- view.  An average of those pixels instead read as a foreign surface: a
-- flat green blob where a canopy has speckle, a colour block where a roof
-- has shingle rows, and "shifted" art where the silhouettes do not line
-- up.  The cap keeps the tile's own UVs (the composite is atlas-sized,
-- block per slot), so geometry and atlas addressing stay shared and the
-- cap can never disagree with the flat render beside it.  The side faces
-- stay on the mid atlas, which is where the facade belongs.
--
-- The composite reads the atlas on the CPU once per slot per rebuild, all
-- behind pcall: a driver that refuses getPixel (or an image too exotic to
-- read) falls back to today's textured top rather than losing the map.

local capWhyWarned = false

--- Make sure this chunk can answer cap faces: builds the atlas-sized
-- composite image on first need and lays down the block for `slot`.
-- Returns true when the slot has cap art, false for a graceful textured
-- fallback.
local function capReady(chunk, slot)
  if chunk.capDead then return false end
  -- Read the atlas on the CPU. love.graphics Images have no getPixel and
  -- canvases no newImageData: the native tileset hands its ImageData
  -- alongside the GPU image (tileset_native keeps `imageData`), which is
  -- also the buffer the palette repaint writes, so it stays in sync.
  local pixels = chunk.pixels
  if pixels == nil then
    local img = chunk.image
    local data = chunk.ts and chunk.ts.imageData
    if not (data and data.getPixel) and img.getPixel then
      data = img
    end
    if not (data and data.getPixel) and img.newImageData then
      local okD, d = pcall(function() return img:newImageData() end)
      data = okD and d or nil
    end
    if not (data and data.getPixel) then
      chunk.capDead = true
      say("cap fallback: no CPU-readable atlas")
      return false
    end
    pixels = data
    chunk.pixels = pixels
  end
  if not chunk.capData then
    local ok, data = pcall(function()
      local iw, ih = pixels:getDimensions()
      return love.image.newImageData(iw, ih)
    end)
    if not ok then
      chunk.capDead = true
      say("cap fallback: no colour image (" .. tostring(data) .. ")")
      return false
    end
    chunk.capData = data
    chunk.capFilled = {}
  end
  if chunk.capFilled[slot] then return true end
  if chunk.capBad and chunk.capBad[slot] then return false end
  local cols = chunk.ts.cols
  local iw, ih = pixels:getDimensions()
  local sx = (slot % cols) * CELL
  local sy = math.floor(slot / cols) * CELL
  if sx + CELL > iw or sy + CELL > ih then
    chunk.capBad = chunk.capBad or {}
    chunk.capBad[slot] = true
    return false
  end
  local odata = chunk.ts and chunk.ts.overImageData
  if odata then
    local okD, ow, oh = pcall(function() return odata:getDimensions() end)
    if not okD or ow ~= iw or oh ~= ih or not odata.getPixel then
      odata = nil
    end
  end
  local okW, errW = pcall(function()
    for dy = 0, CELL - 1 do
      for dx = 0, CELL - 1 do
        local mr, mg, mb, ma = pixels:getPixel(sx + dx, sy + dy)
        local r, g, b, a = mr, mg, mb, ma
        if odata then
          local orr, og, ob, oa = odata:getPixel(sx + dx, sy + dy)
          if oa >= 0.5 then r, g, b, a = orr, og, ob, oa end
        end
        chunk.capData:setPixel(sx + dx, sy + dy, r, g, b, a)
      end
    end
  end)
  if not okW then
    if not capWhyWarned then
      capWhyWarned = true
      say("cap composite refused: " .. tostring(errW))
    end
    chunk.capBad = chunk.capBad or {}
    chunk.capBad[slot] = true
    return false
  end
  chunk.capFilled[slot] = true
  return true
end

-- One quad: four vertices, six indices, in the shared Voxel3D shape.  The
-- quad index is read back off the index list itself, so each chunk counts
-- its own quads.  `skirt` (nil except on a tall side) carries the LOWER
-- neighbour's art top for the bank below the 1:1 span.
local function emitFace(chunk, faceId, x, z, y0, y1, uv, skirt)
  local verts, indices = chunk.verts, chunk.indices
  local corners = Voxel3D.FACE_CORNERS[faceId]
  local au, av = FACE_U[faceId], FACE_V[faceId]
  local shade = Voxel3D.FACE_SHADE[faceId]
  local dy = y1 - y0
  local du, dv = uv.u1 - uv.u0, uv.v1 - uv.v0
  -- one ROW per vertex: love.graphics.newMesh refuses a flat number list
  -- ("expected table of tables"), so rows are what the mesher emits and
  -- what Voxel3D.newMesh passes straight through.
  --
  -- A SIDE face stands the tile UPRIGHT: the atlas is y-down, so the art's
  -- top row (uv.v0) has to land on the face's TOP, not follow the corner
  -- number up from the bottom -- following it drew every facade, trunk and
  -- fence post upside down against the flat field's copy of the same tile.
  -- Top faces keep the direct mapping: their v rides the world's z, which
  -- already runs art-north at the north edge.
  --
  -- A side taller than one tile (a shore face dropping 20px to the water
  -- plane, a map-edge skirt) does not stretch the whole tile over the
  -- span: the top CELL world pixels wear the art 1:1 and the skirt below
  -- wears the lower neighbour's art top rows (the waterline, the path
  -- under a ledge) -- opaque bank material instead of the face's own,
  -- possibly transparent, bottom rows.  Without neighbour art the skirt
  -- extends the face's own bottom row.  The skirt samples row CENTERS,
  -- not the v1 edge: at exactly v1 the nearest sampler rounds into the
  -- next slot's first row, which discards into a black band.  16px sides
  -- are untouched.
  local spans = { { y0, y1, false } }
  if av == 2 and dy > CELL then
    spans = { { y1 - CELL, y1, false }, { y0, y1 - CELL, true } }
  end
  for _, s in ipairs(spans) do
    local b0, b1, isSkirt = s[1], s[2], s[3]
    local sv0, sdvn = uv.v1 - dv / (2 * CELL), 0
    if skirt then sv0, sdvn = skirt.v0, skirt.dvn end
    local shFrac = (b1 - b0) / CELL
    for i = 1, 4 do
      local c = corners[i]
      local yw, v
      if av == 2 then
        local f = c[2]   -- 0 = this quad's bottom, 1 = its top
        yw = b0 + f * (b1 - b0)
        if isSkirt then v = sv0 + (1 - f) * shFrac * sdvn
        else v = uv.v1 - f * dv end
      else
        yw = y0 + c[2] * dy
        v = uv.v0 + c[av] * dv
      end
      verts[#verts + 1] = {
        x + c[1] * CELL,
        yw,
        z + c[3] * CELL,
        uv.u0 + c[au] * du,
        v,
        shade,
      }
    end
    Voxel3D.pushQuad(indices, #indices / 6)
  end
end

local function worldSignature(reachW, reachH)
  local Map = engineRequire("src.core.game3.map")
  if not Map then return "" end
  local s = tostring(Map.current) .. "|" .. reachW .. "," .. reachH
  for i = 1, #Map.world do
    local e = Map.world[i]
    s = s .. ";" .. tostring(e.id) .. "@" .. tostring(e.ox) .. "," .. tostring(e.oy)
  end
  return s
end

-- How the LAST build routed raised tops, kept for capStats(): a zero here
-- with raised cells on screen means the atlas read died and every top
-- silently fell back to raw mid art -- green roofs, untextured canopies.
local capRouted, capWanted = 0, 0

--- Raised tops routed to the composite, out of raised tops wanted.
function FrTerrain.capStats()
  return capRouted, capWanted
end

--- Make sure the mesh matches the world the field is showing.
-- Rebuilds only when the map, the connected-neighbour window or a block
-- itself changed (FrTerrain.markDirty), so walking is free: the vertices
-- live in absolute world coordinates and the camera moves over them.
function FrTerrain.ensure(game, vw, vh)
  local Map = engineRequire("src.core.game3.map")
  if not Map then return false end
  -- the same reach FieldView.draw asks for, so the two never disagree and
  -- recompute the neighbour window every other frame
  local reachW = math.ceil((vw or 0) / CELL)
  local reachH = math.ceil((vh or 0) / CELL)
  Map.refreshWorld(game, reachW, reachH, Map.current)
  local sig = worldSignature(reachW, reachH)
  if not dirty and meshes and sig == signature then return true end

  local maps = game and game.data and game.data.maps
  local rootDef = maps and maps[Map.current]
  local rootL = rootDef and rootDef.midLayout
  if not rootL then
    warnOnce("no metatile layout for " .. tostring(Map.current)
      .. " -- staying on the 2D path")
    return false
  end

  -- The apron shrinks on a map too big to mesh in one piece, so a large
  -- route still gets built (tighter, but built) instead of timing out.
  local apron = APRON
  while apron > 4
      and (rootL.width + 2 * apron) * (rootL.height + 2 * apron) > MAX_CELLS do
    apron = apron - 4
  end
  local x0, y0 = -apron, -apron
  local x1, y1 = rootL.width + apron - 1, rootL.height + apron - 1
  local W, H = x1 - x0 + 1, y1 - y0 + 1
  local world = Map.world

  -- pass one: resolve every cell's tile, height and atlas rectangle
  local cells, atlasOf = {}, {}
  local uvGrid, pairGrid = {}, {}   -- neighbour art lookup for side skirts
  local n, skipped = 0, 0
  for cy = y0, y1 do
    for cx = x0, x1 do
      local def, L, lx, ly = ownerAt(rootDef, rootL, world, cx, cy)
      local mid = L:midAt(lx, ly)
      local coll = L:collAt(lx, ly)
      local y = 0
      if isWater(def, lx, ly, coll) then
        y = -WATER_DROP
      elseif not passable(coll) then
        y = CELL
      end
      scratch[(cy - y0) * W + (cx - x0) + 1] = y

      local pair = def.pair or L.pair or rootDef.pair or "?"
      local chunk = atlasOf[pair]
      if chunk == nil then
        local ts = atlasFor(pair)
        chunk = ts and { pair = pair, image = ts.image, ts = ts,
                         verts = {}, indices = {} } or false
        atlasOf[pair] = chunk
      end
      if chunk then
        local uv, slot = uvFor(chunk.ts, mid)
        if uv then
          n = n + 1
          cells[n] = { cx = cx, cy = cy, y = y, uv = uv, slot = slot,
                       chunk = chunk }
          local gi = (cy - y0) * W + (cx - x0) + 1
          uvGrid[gi], pairGrid[gi] = uv, pair
        else
          skipped = skipped + 1
        end
      else
        skipped = skipped + 1
      end
    end
  end

  if n == 0 then
    if not warned and skipped > 0 then
      warned = true
      say("no native tileset atlas for this map -- staying on the 2D path")
    end
    dirty = false
    signature = sig
    return false
  end
  if skipped > 0 and not warned then
    warned = true
    say(skipped .. " cell(s) had no atlas tile and were left out")
  end

  local function heightAt(cx, cy)
    if cx < x0 or cx > x1 or cy < y0 or cy > y1 then return SKIRT end
    return scratch[(cy - y0) * W + (cx - x0) + 1] or SKIRT
  end

  -- The bank below a tall side face wears the LOWER neighbour's art (the
  -- waterline under a shore face, the path under a ledge): its tile top
  -- rows, 1:1 from the skirt's own top.  nil when the neighbour is off
  -- the meshed window, unmeshed, or from another atlas -- then the skirt
  -- falls back to the face's own bottom row.
  local function skirtFor(nx, ny, pair)
    if nx < x0 or nx > x1 or ny < y0 or ny > y1 then return nil end
    local i = (ny - y0) * W + (nx - x0) + 1
    if pairGrid[i] ~= pair then return nil end
    local nuv = uvGrid[i]
    if not nuv then return nil end
    return { v0 = nuv.v0, dvn = nuv.v1 - nuv.v0 }
  end

  -- pass two: top face, plus whichever side faces a lower neighbour leaves
  capWanted, capRouted = 0, 0
  for i = 1, n do
    local cell = cells[i]
    local cx, cy, y, uv = cell.cx, cell.cy, cell.y, cell.uv
    local chunk = cell.chunk
    local wx, wz = cx * CELL, cy * CELL
    -- a raised cell's top is a CAP (the mid-plus-over composite, see
    -- capReady); ground and anything a cap could not be read keep the
    -- mid texture
    local top = chunk
    if y > 0 then
      capWanted = capWanted + 1
      if capReady(chunk, cell.slot) then
        chunk.cap = chunk.cap or { verts = {}, indices = {} }
        top = chunk.cap
        capRouted = capRouted + 1
      end
    end
    emitFace(top, 3, wx, wz, y, y, uv)
    local e = heightAt(cx + 1, cy)
    if e < y then emitFace(chunk, 1, wx, wz, e, y, uv,
      (y - e > CELL) and skirtFor(cx + 1, cy, chunk.pair) or nil) end
    e = heightAt(cx - 1, cy)
    if e < y then emitFace(chunk, 2, wx, wz, e, y, uv,
      (y - e > CELL) and skirtFor(cx - 1, cy, chunk.pair) or nil) end
    e = heightAt(cx, cy + 1)
    if e < y then emitFace(chunk, 5, wx, wz, e, y, uv,
      (y - e > CELL) and skirtFor(cx, cy + 1, chunk.pair) or nil) end
    e = heightAt(cx, cy - 1)
    if e < y then emitFace(chunk, 6, wx, wz, e, y, uv,
      (y - e > CELL) and skirtFor(cx, cy - 1, chunk.pair) or nil) end
  end

  say(("caps: %d of %d raised tops routed to art"):format(
    capRouted, capWanted))

  local out, count = {}, 0
  for _, chunk in pairs(atlasOf) do
    if chunk then
      local mesh = Voxel3D.newMesh(chunk.verts, chunk.indices)
      if mesh then
        local entry = { mesh = mesh, image = chunk.image }
        if chunk.cap and chunk.capData then
          local okI, capImage = pcall(function()
            return love.graphics.newImage(chunk.capData)
          end)
          if okI then
            local capMesh = Voxel3D.newMesh(chunk.cap.verts,
                                            chunk.cap.indices)
            if capMesh then
              capImage:setFilter("nearest", "nearest")
              entry.capMesh, entry.capImage = capMesh, capImage
              say("cap mesh ready for " .. tostring(chunk.pair) .. " ("
                .. #chunk.cap.indices / 6 .. " faces)")
            else
              say("cap fallback: mesh refused for " .. tostring(chunk.pair))
            end
          else
            say("cap fallback: newImage " .. tostring(capImage))
          end
        elseif chunk.cap then
          say("cap fallback: colour data missing for "
            .. tostring(chunk.pair))
        end
        out[chunk.pair] = entry
        count = count + 1
      end
    end
  end
  if count == 0 then
    warnOnce(("no mesh built for %d cell(s) -- the driver refused the "
      .. "vertex format, staying on the 2D path"):format(n))
    return false
  end

  FrTerrain.release()
  meshes = out
  heights, scratch = scratch, heights
  rect = { x0 = x0, y0 = y0, W = W }
  signature = sig
  dirty = false
  return true
end

--- Ground height under a world pixel, for anchoring a character to the
-- terrain it is standing on.  Returns 0 outside the built rectangle.
function FrTerrain.heightAtWorld(wx, wy)
  if not rect then return 0 end
  local cx = math.floor((wx or 0) / CELL)
  local cy = math.floor((wy or 0) / CELL)
  local i = (cy - rect.y0) * rect.W + (cx - rect.x0) + 1
  local h = heights[i]
  if h == nil then return 0 end
  return h
end

--- A block was replaced on the map (Cut, Rock Smash, a script): the
-- collision byte under it changed, so the extrusion has to follow.
function FrTerrain.markDirty()
  dirty = true
end

function FrTerrain.release()
  if meshes then
    for _, e in pairs(meshes) do
      if e.mesh and e.mesh.release then pcall(e.mesh.release, e.mesh) end
      if e.capMesh and e.capMesh.release then
        pcall(e.capMesh.release, e.capMesh)
      end
    end
    meshes = nil
  end
  rect = nil
  dirty = true
end

FrTerrain.invalidate = FrTerrain.release

--- Draw every built chunk under the current 3D camera.
function FrTerrain.draw()
  if not meshes then return false end
  local drew = false
  for _, e in pairs(meshes) do
    if e.mesh then
      Voxel3D.draw(e.mesh, e.image, nil, 0)
      drew = true
    end
    -- the cap pass shares depth with the terrain it sits on: same scene,
    -- same camera, same pull (0) -- z resolves the coplanar edges
    if e.capMesh then
      Voxel3D.draw(e.capMesh, e.capImage, nil, 0)
    end
  end
  return drew
end

--- Draw every built chunk into the sun's shadow map (ShadowMap's pass).
-- Same geometry as draw(), minus camera business: model is identity
-- because the meshes already live in world coordinates, and there is no
-- pull (a shadow must not ride toward the camera).
function FrTerrain.cast()
  local ShadowMap = V.require("ShadowMap")
  if not meshes then return end
  for _, e in pairs(meshes) do
    if e.mesh then ShadowMap.draw(e.mesh, e.image, nil) end
    if e.capMesh then ShadowMap.draw(e.capMesh, e.capImage, nil) end
  end
end

--- A stamp of everything that changes what the shadow map would hold:
-- the world signature that rebuilt the meshes, plus the dirty flag (a
-- rebuild pending means the meshes are about to change).  ShadowMap's
-- staleness check compares this to decide whether to recast.
function FrTerrain.stateSig()
  return (signature or "") .. (dirty and "d" or "")
end

function FrTerrain.ready()
  return meshes ~= nil
end

return FrTerrain
