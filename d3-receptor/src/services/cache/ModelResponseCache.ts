import { GenerationResponse, ModelTask } from '../models/ModelService';

interface CacheEntry {
  response: GenerationResponse;
  timestamp: number;
  expiresAt: number;
}

interface CacheOptions {
  ttl?: number;  // Time-to-live in milliseconds
  maxEntries?: number;
}

/**
 * Provides caching for model responses to reduce API calls and improve performance
 */
export class ModelResponseCache {
  private cache: Map<string, CacheEntry> = new Map();
  private ttl: number; // Default cache TTL in milliseconds (1 hour)
  private maxEntries: number; // Maximum number of entries to keep in the cache

  constructor(options: CacheOptions = {}) {
    this.ttl = options.ttl || 3600000; // 1 hour default TTL
    this.maxEntries = options.maxEntries || 100; // Default maximum entries
  }

  /**
   * Generate a cache key from the prompt, task, and model provider
   */
  private generateKey(prompt: string, task: ModelTask, provider?: string): string {
    return `${task}:${provider || 'default'}:${prompt}`;
  }

  /**
   * Store a response in the cache
   */
  set(prompt: string, task: ModelTask, response: GenerationResponse, provider?: string): void {
    const key = this.generateKey(prompt, task, provider);
    const now = Date.now();
    
    // Add to cache with timestamp and expiration
    this.cache.set(key, {
      response,
      timestamp: now,
      expiresAt: now + this.ttl
    });

    // If cache exceeds maximum size, remove oldest entries
    if (this.cache.size > this.maxEntries) {
      this.evictOldest();
    }
  }

  /**
   * Get a response from the cache if it exists and is not expired
   */
  get(prompt: string, task: ModelTask, provider?: string): GenerationResponse | null {
    const key = this.generateKey(prompt, task, provider);
    const entry = this.cache.get(key);
    
    if (!entry) {
      return null;
    }
    
    // Check if the entry is expired
    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return null;
    }
    
    return entry.response;
  }

  /**
   * Check if a response exists in the cache and is not expired
   */
  has(prompt: string, task: ModelTask, provider?: string): boolean {
    const key = this.generateKey(prompt, task, provider);
    const entry = this.cache.get(key);
    
    if (!entry) {
      return false;
    }
    
    // Check if the entry is expired
    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return false;
    }
    
    return true;
  }

  /**
   * Clear all entries from the cache
   */
  clear(): void {
    this.cache.clear();
  }

  /**
   * Evict expired entries from the cache
   */
  evictExpired(): void {
    const now = Date.now();
    
    for (const [key, entry] of this.cache.entries()) {
      if (now > entry.expiresAt) {
        this.cache.delete(key);
      }
    }
  }

  /**
   * Evict the oldest entries from the cache to make room for new ones
   */
  private evictOldest(): void {
    // Get all entries sorted by timestamp (oldest first)
    const entries = Array.from(this.cache.entries())
      .sort(([, a], [, b]) => a.timestamp - b.timestamp);
    
    // Remove the oldest 10% or at least one entry
    const removeCount = Math.max(1, Math.floor(entries.length * 0.1));
    
    for (let i = 0; i < removeCount; i++) {
      if (entries[i]) {
        this.cache.delete(entries[i][0]);
      }
    }
  }
} 