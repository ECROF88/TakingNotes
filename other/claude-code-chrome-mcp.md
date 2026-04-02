Solution for chrome mcp of claude code in WSL2


# Install
claude mcp add-json chrome-devtools '{"type":"stdio","command":"cmd.exe","args":["/c", "npx", "-y", "chrome-devtools-mcp@latest","--isolated", "true", "--chrome-arg=--disable-extensions", "--chrome-arg=--no-first-run", "--chrome-arg=--no-first-run"],"env":{}}'

# Test:
claude mcp get chrome-devtools 
claude mcp list

# Remove
claude mcp remove chrome-devtools

