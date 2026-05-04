/**
 * BlueprintBid Local LLM Provider
 *
 * Drop-in replacement for the OpenAI vision-provider.
 * Calls the local (or Modal-deployed) LLM server to analyze
 * construction drawing images.
 *
 * Usage in your Next.js API route:
 *   import { extractWithLocalLLM } from '@/lib/document-ai/local-llm-provider';
 *   const result = await extractWithLocalLLM(imageBuffer, ocrText, 'millwork');
 */

// ── Types (match your existing lib/document-ai/types.ts) ─────────────

export interface ScopeItem {
  id: string;
  category: string;
  description: string;
  quantity: number;
  unit: 'ea' | 'lf' | 'sf' | 'lot';
  confidence: number;
  pageRefs: number[];
  textSnippets: string[];
}

export interface SpecSheetEntry {
  key: string;
  value: string;
  confidence: number;
  pageRefs: number[];
  textSnippets: string[];
}

export interface Specification {
  finish_code: string;
  finish_name: string;
  category: string;
  manufacturer: string;
  description: string;
  finish: string;
}

export interface ExtractionResult {
  scopeItems: ScopeItem[];
  specSheet: SpecSheetEntry[];
  specifications: Specification[];
}

// ── Configuration ────────────────────────────────────────────────────

const LLM_BASE_URL = process.env.LOCAL_LLM_URL ?? 'http://localhost:8000';
const LLM_API_KEY = process.env.LOCAL_LLM_API_KEY ?? '';
const LLM_TIMEOUT_MS = parseInt(process.env.LOCAL_LLM_TIMEOUT_MS ?? '60000', 10);

// Modal deployment URL (if using serverless GPU)
const MODAL_URL = process.env.MODAL_LLM_URL ?? '';
const MODAL_TOKEN = process.env.MODAL_TOKEN ?? '';

// ── Local server provider (multipart) ────────────────────────────────

export async function extractWithLocalLLM(
  imageBuffer: Buffer,
  text: string,
  trade: string,
): Promise<ExtractionResult> {
  const formData = new FormData();
  const blob = new Blob([imageBuffer], { type: 'image/png' });
  formData.append('image', blob, 'page.png');
  formData.append('text', text);
  formData.append('trade', trade);

  const headers: Record<string, string> = {};
  if (LLM_API_KEY) {
    headers['Authorization'] = `Bearer ${LLM_API_KEY}`;
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), LLM_TIMEOUT_MS);

  try {
    const response = await fetch(`${LLM_BASE_URL}/analyze-drawing`, {
      method: 'POST',
      headers,
      body: formData,
      signal: controller.signal,
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`LLM server error ${response.status}: ${errText}`);
    }

    const result: ExtractionResult = await response.json();
    return validateResult(result);
  } finally {
    clearTimeout(timeout);
  }
}

// ── Modal serverless provider (JSON/base64) ──────────────────────────

export async function extractWithModal(
  imageBuffer: Buffer,
  text: string,
  trade: string,
): Promise<ExtractionResult> {
  const url = MODAL_URL;
  if (!url) {
    throw new Error('MODAL_LLM_URL not configured');
  }

  const imageBase64 = imageBuffer.toString('base64');

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (MODAL_TOKEN) {
    headers['Authorization'] = `Bearer ${MODAL_TOKEN}`;
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), LLM_TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        image_base64: imageBase64,
        text,
        trade,
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`Modal inference error ${response.status}: ${errText}`);
    }

    const result: ExtractionResult = await response.json();
    return validateResult(result);
  } finally {
    clearTimeout(timeout);
  }
}

// ── Hybrid provider (local → Modal → OpenAI fallback) ────────────────

export async function extractWithHybrid(
  imageBuffer: Buffer,
  text: string,
  trade: string,
  openAIFallback?: (buf: Buffer, text: string, trade: string) => Promise<ExtractionResult>,
): Promise<ExtractionResult> {
  // Try local LLM first
  try {
    console.log('[HYBRID] Attempting local LLM …');
    return await extractWithLocalLLM(imageBuffer, text, trade);
  } catch (localErr) {
    console.warn('[HYBRID] Local LLM failed:', localErr);
  }

  // Try Modal
  if (MODAL_URL) {
    try {
      console.log('[HYBRID] Attempting Modal …');
      return await extractWithModal(imageBuffer, text, trade);
    } catch (modalErr) {
      console.warn('[HYBRID] Modal failed:', modalErr);
    }
  }

  // Fall back to OpenAI
  if (openAIFallback) {
    console.log('[HYBRID] Falling back to OpenAI …');
    return await openAIFallback(imageBuffer, text, trade);
  }

  throw new Error('All inference providers failed');
}

// ── Health check ─────────────────────────────────────────────────────

export async function checkHealth(): Promise<{
  status: string;
  model: string;
  gpu_available: boolean;
}> {
  const response = await fetch(`${LLM_BASE_URL}/health`, {
    method: 'GET',
    signal: AbortSignal.timeout(5000),
  });
  return response.json();
}

// ── Validation helper ────────────────────────────────────────────────

function validateResult(raw: ExtractionResult): ExtractionResult {
  return {
    scopeItems: (raw.scopeItems ?? []).map((item) => ({
      id: item.id ?? crypto.randomUUID(),
      category: item.category ?? 'General',
      description: item.description ?? '',
      quantity: Number(item.quantity) || 0,
      unit: (['ea', 'lf', 'sf', 'lot'].includes(item.unit) ? item.unit : 'ea') as ScopeItem['unit'],
      confidence: Math.min(1, Math.max(0, Number(item.confidence) || 0)),
      pageRefs: Array.isArray(item.pageRefs) ? item.pageRefs : [1],
      textSnippets: Array.isArray(item.textSnippets) ? item.textSnippets : [],
    })),
    specSheet: (raw.specSheet ?? []).map((entry) => ({
      key: entry.key ?? '',
      value: entry.value ?? '',
      confidence: Math.min(1, Math.max(0, Number(entry.confidence) || 0)),
      pageRefs: Array.isArray(entry.pageRefs) ? entry.pageRefs : [1],
      textSnippets: Array.isArray(entry.textSnippets) ? entry.textSnippets : [],
    })),
    specifications: raw.specifications ?? [],
  };
}
