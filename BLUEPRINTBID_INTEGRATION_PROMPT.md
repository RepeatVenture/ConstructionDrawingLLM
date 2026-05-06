# BlueprintBid Integration Prompt

**Copy this entire prompt and give it to the AI assistant working in your BlueprintBid repository:**

---

## Task: Replace OpenAI Vision API with Custom LLaVA API

I have deployed a custom fine-tuned LLaVA vision-language model for construction drawing analysis. I need to integrate it into my BlueprintBid Next.js application by replacing the current OpenAI API calls.

## Deployed API Details

**Base URL:**
```
https://repeatventure--llava-construction-api-fastapi-app.modal.run
```

**Available Endpoints:**
- `GET /` - API info
- `GET /health` - Health check
- `GET /info` - Model information
- `POST /analyze` - Main inference endpoint (this is what we'll use)

## Current Setup

The BlueprintBid app currently uses OpenAI's GPT-4 Vision API to analyze construction drawings and extract millwork items. I need you to:

1. **Find** all OpenAI API calls related to construction drawing analysis
2. **Replace** them with calls to my custom LLaVA API
3. **Update** environment variables
4. **Ensure** the response format is handled correctly

## API Request Format

### OpenAI (Current - BEFORE):
```typescript
const response = await openai.chat.completions.create({
  model: "gpt-4-vision-preview",
  messages: [
    {
      role: "system",
      content: systemPrompt
    },
    {
      role: "user",
      content: [
        {
          type: "text",
          text: userPrompt
        },
        {
          type: "image_url",
          image_url: {
            url: `data:image/png;base64,${imageBase64}`
          }
        }
      ]
    }
  ],
  max_tokens: 4000,
  temperature: 0.7,
  response_format: { type: "json_object" }
});

const content = response.choices[0].message.content;
const data = JSON.parse(content);
```

### LLaVA API (New - AFTER):
```typescript
const response = await fetch(`${process.env.LLAVA_API_URL}/analyze`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    image: `data:image/png;base64,${imageBase64}`,
    text: userPrompt,
    system_prompt: systemPrompt,
    max_tokens: 4000,
    temperature: 0.7,
    response_format: "json"
  }),
  signal: AbortSignal.timeout(120000) // 2 minute timeout
});

if (!response.ok) {
  throw new Error(`LLaVA API error: ${response.status} ${response.statusText}`);
}

const result = await response.json();
const data = JSON.parse(result.content); // Content is JSON string
```

## API Response Format

The LLaVA API returns:
```typescript
{
  "content": "JSON string containing the extraction results",
  "raw_output": "Original model output",
  "model": "llava-construction-v1",
  "usage": {
    "prompt_tokens": 1234,
    "completion_tokens": 567,
    "total_tokens": 1801
  }
}
```

**Important:** The `content` field is a **JSON string** that needs to be parsed with `JSON.parse()`.

## Expected Data Structure

After parsing `result.content`, you should get:
```typescript
{
  "included_items": [
    {
      "room_number": "101",
      "room_name": "Living Room",
      "drawing_refs": ["A2.1", "A2.2"],
      "category": "cabinet",
      "description": "Base cabinet with drawers",
      "finish_code": "WO-1",
      "count": 2,
      "quantity": 4,
      "unit": "LF",
      "confidence": 0.9
    }
    // ... more items
  ],
  "specifications": [
    {
      "key": "finish",
      "value": "Natural oak",
      "confidence": 0.85
    }
    // ... more specs
  ],
  "notes": "Optional notes or errors"
}
```

## Environment Variables

Add to Vercel (or `.env.local` for local development):

```env
# LLaVA API Configuration
LLAVA_API_URL=https://repeatventure--llava-construction-api-fastapi-app.modal.run
LLAVA_API_TIMEOUT=120000

# Keep OpenAI as fallback (optional)
# OPENAI_API_KEY=sk-...
```

## Step-by-Step Integration Tasks

### 1. Find OpenAI Integration

**Search for:**
- Files importing `openai` package
- API calls to OpenAI (likely in `lib/`, `utils/`, or `api/` directories)
- Functions handling construction drawing analysis
- Look for keywords: `gpt-4-vision`, `openai.chat.completions`, `vision`, `millwork`, `construction drawing`

**Typical file locations to check:**
- `lib/openai.ts` or `lib/api.ts`
- `app/api/analyze/route.ts` (Next.js 13+ App Router)
- `pages/api/analyze.ts` (Next.js Pages Router)
- `utils/analyzedrawing.ts` or similar

### 2. Create LLaVA API Client

Create a new file (e.g., `lib/llava-api.ts`):

```typescript
interface AnalyzeRequest {
  image: string; // base64 data URI
  text: string;
  system_prompt?: string;
  max_tokens?: number;
  temperature?: number;
}

interface AnalyzeResponse {
  content: string; // JSON string
  model: string;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export async function analyzeDrawing(
  imageBase64: string,
  prompt: string,
  systemPrompt: string = "You are an expert at reading construction drawings and extracting millwork specifications."
): Promise<any> {
  const apiUrl = process.env.LLAVA_API_URL || process.env.NEXT_PUBLIC_LLAVA_API_URL;
  
  if (!apiUrl) {
    throw new Error("LLAVA_API_URL environment variable is not set");
  }

  // Ensure image has data URI prefix
  const imageDataUri = imageBase64.startsWith('data:') 
    ? imageBase64 
    : `data:image/png;base64,${imageBase64}`;

  const request: AnalyzeRequest = {
    image: imageDataUri,
    text: prompt,
    system_prompt: systemPrompt,
    max_tokens: 4000,
    temperature: 0.7,
    response_format: "json"
  };

  try {
    const response = await fetch(`${apiUrl}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
      signal: AbortSignal.timeout(120000), // 2 minute timeout
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`LLaVA API error: ${response.status} - ${errorText}`);
    }

    const result: AnalyzeResponse = await response.json();
    
    // Parse the JSON string content
    try {
      return JSON.parse(result.content);
    } catch (parseError) {
      console.error('Failed to parse LLaVA response:', result.content);
      throw new Error('Invalid JSON response from LLaVA API');
    }
  } catch (error) {
    if (error instanceof Error) {
      if (error.name === 'AbortError' || error.message.includes('timeout')) {
        throw new Error('LLaVA API request timed out. The model may be cold-starting (first request takes ~5 minutes).');
      }
      throw error;
    }
    throw new Error('Unknown error calling LLaVA API');
  }
}

// Health check function
export async function checkLLaVAHealth(): Promise<boolean> {
  const apiUrl = process.env.LLAVA_API_URL || process.env.NEXT_PUBLIC_LLAVA_API_URL;
  
  if (!apiUrl) return false;
  
  try {
    const response = await fetch(`${apiUrl}/health`, { 
      signal: AbortSignal.timeout(5000) 
    });
    return response.ok;
  } catch {
    return false;
  }
}
```

### 3. Replace OpenAI Calls

In your analysis endpoint/function, replace:

```typescript
// BEFORE (OpenAI)
import OpenAI from 'openai';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

const response = await openai.chat.completions.create({
  model: "gpt-4-vision-preview",
  messages: [...],
  // ...
});

const data = JSON.parse(response.choices[0].message.content);
```

```typescript
// AFTER (LLaVA)
import { analyzeDrawing } from '@/lib/llava-api';

const data = await analyzeDrawing(
  imageBase64,
  userPrompt,
  systemPrompt
);
```

### 4. Update Error Handling

Add specific error handling for cold starts:

```typescript
try {
  const data = await analyzeDrawing(imageBase64, prompt, systemPrompt);
  return data;
} catch (error) {
  if (error instanceof Error && error.message.includes('cold-starting')) {
    // Inform user that first request takes longer
    return {
      error: 'Model is starting up (first request ~5 minutes). Please try again in a moment.',
      retry: true
    };
  }
  throw error;
}
```

### 5. Add Loading State

Since cold starts take time, update UI:

```typescript
// In your component
const [isAnalyzing, setIsAnalyzing] = useState(false);
const [statusMessage, setStatusMessage] = useState('');

const handleAnalyze = async () => {
  setIsAnalyzing(true);
  setStatusMessage('Analyzing drawing... (First request may take 5-10 minutes)');
  
  try {
    const result = await analyzeDrawing(image, prompt, systemPrompt);
    // Handle result
  } catch (error) {
    // Handle error
  } finally {
    setIsAnalyzing(false);
    setStatusMessage('');
  }
};
```

### 6. Testing Checklist

- [ ] Environment variables set in Vercel
- [ ] LLaVA API health check passes
- [ ] Can upload and analyze a construction drawing
- [ ] Results match expected schema (included_items, specifications)
- [ ] Error handling works (timeouts, invalid responses)
- [ ] Loading states show appropriate messages
- [ ] Cold start warning displayed on first use

## Key Differences to Note

| Feature | OpenAI | LLaVA API |
|---------|---------|-----------|
| **Request format** | `messages` array | Flat `image`, `text`, `system_prompt` |
| **Image format** | Can be URL or base64 | Base64 data URI required |
| **Response** | `choices[0].message.content` | `result.content` (as JSON string) |
| **Timeout** | ~30 seconds typical | 2 minutes (120s) recommended |
| **Cold start** | None | 5-10 minutes first request |
| **Cost** | Per token | Per second of GPU usage |

## Common Issues & Solutions

### Issue: "Cannot find module '@/lib/llava-api'"
**Solution:** Adjust import path based on your project structure. May be `'lib/llava-api'` or `'@/utils/llava-api'`.

### Issue: Request times out
**Solution:** 
- First request takes 5-10 minutes (model downloading)
- Increase timeout to 120 seconds minimum
- Show loading message to users

### Issue: Response parsing fails
**Solution:**
- The API returns `content` as a JSON string
- Always use `JSON.parse(result.content)`
- Add try-catch around parsing

### Issue: CORS errors in client-side
**Solution:**
- The API has CORS enabled for all origins
- Make API calls from server-side (API routes) not client-side
- Use Next.js API routes as proxy if needed

## Verification

After integration, verify:

1. **Health check works:**
   ```bash
   curl https://repeatventure--llava-construction-api-fastapi-app.modal.run/health
   ```

2. **Test extraction** with a known drawing
3. **Compare results** with previous OpenAI output (should be similar structure)
4. **Check logs** in Vercel for any errors
5. **Monitor costs** in Modal dashboard: https://modal.com/apps

## Support Information

- **API Dashboard:** https://modal.com/apps/repeatventure/main/deployed/llava-construction-api
- **Model:** LLaVA-v1.6-Vicuna-7B fine-tuned on 519 construction drawing samples
- **Cold start:** First request takes 5-10 minutes, then fast (~5-10 seconds)
- **Scaling:** Auto-scales to zero when idle (no cost)
- **Cost:** ~$0.05 per image analysis (8 seconds @ $0.00060/sec on T4 GPU)

## Summary

**What you need to do:**
1. Add environment variable `LLAVA_API_URL` to Vercel
2. Create `lib/llava-api.ts` with the provided client code
3. Find and replace OpenAI calls with `analyzeDrawing()` function
4. Update error handling and loading states
5. Test with real construction drawings

The integration should be straightforward - it's essentially swapping one API client for another with minimal changes to business logic.

---

**After you complete the integration, please:**
- Confirm all OpenAI calls have been replaced
- Test with at least 3 different construction drawings
- Report any issues with response format or parsing
- Share before/after comparison if results differ significantly
