-- Inline code in manuscript.md is mostly file paths and identifiers. Typeset it
-- with xurl's \nolinkurl so long paths can break at any character instead of
-- running into the margin. Code containing spaces (\url drops them) or characters
-- \url cannot take inside another command's argument falls back to pandoc's
-- default \texttt.

local function balanced(s)
  local depth = 0
  for c in s:gmatch("[{}]") do
    depth = depth + (c == "{" and 1 or -1)
    if depth < 0 then return false end
  end
  return depth == 0
end

function Code(el)
  local s = el.text
  if s:find("[%%#\\%s]") or not balanced(s) then
    return nil
  end
  return pandoc.RawInline("latex", "\\nolinkurl{" .. s .. "}")
end
