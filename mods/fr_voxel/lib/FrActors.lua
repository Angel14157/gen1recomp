-- Characters in the FireRed diorama.
--
-- There is no voxel character model here: a FireRed sprite is 16x16 of
-- art that only exists as art, so it is drawn the way the flat field
-- draws it -- through the engine's own actor path, with the same fallback
-- chain (live sprite sheet, resolved graphics id, finally a block) -- and
-- the camera is slid so the projected anchor and the flat draw coincide.
-- The depth scale then grows the figure with its distance from the eye,
-- which is the one thing the flat path cannot do.
--
-- Actors are composited after the depth-tested terrain (Voxel3D's overlay
-- pass), so a character never disappears behind geometry: honest occlusion
-- needs the sprite slabs and ghost pass the Dramatic Shape voxel mod
-- builds, which is a FASE 3 item.  Draw order inside the overlay is the
-- field's own: normal-priority actors first, then the elevated ones.

local V = ...
local Voxel3D = V.require("Voxel3D")
local FrTerrain = V.require("FrTerrain")

local FrActors = {}

local function fieldView()
  local ok, F = pcall(require, "src.core.game3.field_view")
  if ok and type(F) == "table" and F.drawActor then return F end
  return nil
end

--- Draw every collected actor into the scene canvas.
-- `cast` is FieldView.pipelineActors' result; `ctx.state` is the game.
-- `w, h` is the scene canvas in framebuffer pixels: the flat path draws
-- its sprites in WORLD pixels and lets the viewport blit stretch them to
-- the screen, while this overlay draws straight into the full-resolution
-- canvas -- so without the world->canvas factor here a 16px sprite lands
-- as 16 canvas pixels, a few times too small on any screen bigger than
-- 240x160.  The projection supplies the depth half (k); the framing
-- supplies the rest, exactly as viewProjection maps the terrain mesh.
function FrActors.draw(ctx, cast, w, h)
  if not (ctx and cast) then return end
  local F = fieldView()
  if not F then return end
  local vw = tonumber(ctx.vw) or 240
  local vh = tonumber(ctx.vh) or 160
  w, h = tonumber(w), tonumber(h)
  if not (w and h and w > 0 and h > 0) then
    if love.graphics.getPixelDimensions then
      w, h = love.graphics.getPixelDimensions()
    else
      w, h = 240, 160
    end
  end
  -- canvas pixels per world pixel at the focus plane: viewProjection
  -- frames vw x vh there and divides clip by w, so both axes come out
  -- of the canvas and the view, in that order
  local bx, by = w / vw, h / vh
  local list = {}
  for _, a in ipairs(cast.under or {}) do list[#list + 1] = a end
  for _, a in ipairs(cast.over or {}) do list[#list + 1] = a end

  love.graphics.setColor(1, 1, 1, 1)
  for i = 1, #list do
    local a = list[i]
    local fx = (a.x or 0) + 8
    local fy = (a.y or 0) + 16
    local ground = FrTerrain.heightAtWorld(fx, fy)
    local sx, sy, k = Voxel3D.project(fx, ground, fy)
    if sx and sy then
      k = tonumber(k) or 1
      if k < 0.05 then k = 0.05 elseif k > 8 then k = 8 end
      local tx, ty = k * bx, k * by
      love.graphics.push()
      love.graphics.scale(tx, ty)
      -- slide the flat camera so this actor's feet land on (sx/tx, sy/ty),
      -- which the scale above puts back on (sx, sy)
      F.drawActor(ctx.state, cast.mapDef, a, fx - sx / tx, fy - sy / ty)
      love.graphics.pop()
    end
  end
  love.graphics.setColor(1, 1, 1, 1)
end

return FrActors
