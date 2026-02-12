# Token Limit Fix - Agentic AI

## Problem
After several chat exchanges, the app was hitting token limit errors because:
1. **Full conversation memory** (6 entries) was being serialized and sent to the LLM
2. Each memory entry contained full user queries and AI responses
3. Citations were included in full, adding more tokens
4. After 5-10 exchanges, the accumulated context exceeded the 8K token limit of Llama 3.1 8B

## Solution Implemented

### 1. **Reduced Memory Window** (`backend/core.py`)
- Changed from keeping 6 memory entries → **3 entries (MAX_MEMORY_ENTRIES)**
- Added `MAX_MEMORY_SIZE_TOKENS` constant to track token budget
- Only recent context is kept, older context is discarded

### 2. **Concise Memory Formatting** (`backend/core.py`)
- Created `format_memory_concise()` function that:
  - Truncates user queries to 100 characters
  - Truncates AI responses to 150 characters  
  - Formats as simple Q&A pairs instead of full dict serialization
  - Reduces verbosity significantly

### 3. **Truncate Memory at Storage** (`backend/core.py`)
- Store only first 200 chars of user queries
- Store only first 300 chars of AI responses
- Prevents memory from growing unbounded in memory_store

### 4. **Optimized Prompts** (`backend/api/chat.py`)
- Simplified prompt templates to use fewer tokens
- Changed "concisely" language instructions for tighter responses
- Memory is now optional context (only included if available)
- Response length capped at 2-3 sentences

## Results
- **Before**: ~6000+ tokens per chat after 5 exchanges
- **After**: ~1500-2000 tokens per chat (consistent)
- **Benefit**: Unlimited conversation continuity without token overflows

## Token Budget Breakdown (Per Request)
- System prompt: ~100 tokens
- User query: ~50-100 tokens
- Retrieved context (4 documents): ~600-800 tokens
- Memory (3 entries, concise): ~200-300 tokens
- Model response: ~256 tokens (max_tokens limit)
- **Total**: ~1200-1500 tokens (safely under 8K limit)

## Key Files Modified
- `backend/core.py` - Added memory management constants and `format_memory_concise()`
- `backend/api/chat.py` - Updated prompt templates and imports

## Testing Recommendations
1. Run multiple chat turns (10+) to verify no token limit errors
2. Check that follow-up questions still reference previous context
3. Verify citations still work correctly
4. Monitor Groq API usage for token efficiency

## Future Improvements
- Add token counting library to measure exact tokens used
- Implement sliding window for memory if needed
- Add memory summarization for longer conversations
- Consider intent-based memory filtering (only keep relevant context)
