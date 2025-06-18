use std::collections::{HashMap, BTreeMap};
use serde_json::Value;
use crate::chigutiro::{ChigutiroResult, ChigutiroError};

/// Sorted index for range queries using a B-tree
pub struct SortedIndex {
    /// B-tree mapping values to record IDs
    index: BTreeMap<Value, Vec<u64>>,
    
    /// Whether the index is optimized
    optimized: bool,
}

impl SortedIndex {
    /// Create a new sorted index
    pub fn new() -> Self {
        Self {
            index: BTreeMap::new(),
            optimized: false,
        }
    }
    
    /// Add a record to the index
    pub fn add(&mut self, record_id: u64, value: &Value) -> ChigutiroResult<()> {
        self.index
            .entry(value.clone())
            .or_insert_with(Vec::new)
            .push(record_id);
        
        self.optimized = false; // Mark as needing optimization
        Ok(())
    }
    
    /// Perform a range query
    pub fn range_query(&self, min: &Value, max: &Value) -> ChigutiroResult<Vec<u64>> {
        let mut result = Vec::new();
        
        for (value, record_ids) in self.index.range(min.clone()..=max.clone()) {
            result.extend(record_ids);
        }
        
        Ok(result)
    }
    
    /// Get all records with a specific value
    pub fn get_records(&self, value: &Value) -> ChigutiroResult<Vec<u64>> {
        Ok(self.index.get(value).cloned().unwrap_or_default())
    }
    
    /// Optimize the index
    pub fn optimize(&mut self) -> ChigutiroResult<()> {
        if !self.optimized {
            // Sort record ID vectors for better cache locality
            for record_ids in self.index.values_mut() {
                record_ids.sort_unstable();
                record_ids.dedup(); // Remove duplicates
            }
            self.optimized = true;
        }
        Ok(())
    }
    
    /// Get index statistics
    pub fn stats(&self) -> IndexStats {
        let unique_values = self.index.len();
        let total_entries: usize = self.index.values().map(|v| v.len()).sum();
        
        IndexStats {
            unique_values,
            total_entries,
            memory_usage_estimate: std::mem::size_of_val(&self.index) + 
                                  total_entries * std::mem::size_of::<u64>(),
            optimized: self.optimized,
        }
    }
}

/// Hash index for exact matches
pub struct HashIndex {
    /// Hash map from values to record IDs
    index: HashMap<Value, Vec<u64>>,
    
    /// Whether the index is optimized
    optimized: bool,
}

impl HashIndex {
    /// Create a new hash index
    pub fn new() -> Self {
        Self {
            index: HashMap::new(),
            optimized: false,
        }
    }
    
    /// Add a record to the index
    pub fn add(&mut self, record_id: u64, value: &Value) -> ChigutiroResult<()> {
        self.index
            .entry(value.clone())
            .or_insert_with(Vec::new)
            .push(record_id);
        
        self.optimized = false;
        Ok(())
    }
    
    /// Get all records with a specific value
    pub fn get_records(&self, value: &Value) -> ChigutiroResult<Vec<u64>> {
        Ok(self.index.get(value).cloned().unwrap_or_default())
    }
    
    /// Check if a value exists in the index
    pub fn contains(&self, value: &Value) -> bool {
        self.index.contains_key(value)
    }
    
    /// Optimize the index
    pub fn optimize(&mut self) -> ChigutiroResult<()> {
        if !self.optimized {
            // Sort and deduplicate record ID vectors
            for record_ids in self.index.values_mut() {
                record_ids.sort_unstable();
                record_ids.dedup();
            }
            
            // Shrink to fit to reduce memory usage
            self.index.shrink_to_fit();
            self.optimized = true;
        }
        Ok(())
    }
    
    /// Get index statistics
    pub fn stats(&self) -> IndexStats {
        let unique_values = self.index.len();
        let total_entries: usize = self.index.values().map(|v| v.len()).sum();
        
        IndexStats {
            unique_values,
            total_entries,
            memory_usage_estimate: std::mem::size_of_val(&self.index) + 
                                  total_entries * std::mem::size_of::<u64>(),
            optimized: self.optimized,
        }
    }
}

/// Bloom filter index for probabilistic membership testing
pub struct BloomIndex {
    /// Bit array
    bits: Vec<u8>,
    
    /// Number of bits in the array
    num_bits: usize,
    
    /// Number of hash functions
    num_hashes: usize,
    
    /// Number of items added
    items_added: usize,
    
    /// Expected false positive rate
    false_positive_rate: f64,
}

impl BloomIndex {
    /// Create a new bloom filter index
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        let num_bits = Self::calculate_optimal_bits(expected_items, false_positive_rate);
        let num_hashes = Self::calculate_optimal_hashes(num_bits, expected_items);
        
        Self {
            bits: vec![0; (num_bits + 7) / 8], // Round up to nearest byte
            num_bits,
            num_hashes,
            items_added: 0,
            false_positive_rate,
        }
    }
    
    /// Calculate optimal number of bits
    fn calculate_optimal_bits(items: usize, fpr: f64) -> usize {
        let ln2_squared = std::f64::consts::LN_2 * std::f64::consts::LN_2;
        ((-1.0 * items as f64 * fpr.ln()) / ln2_squared).ceil() as usize
    }
    
    /// Calculate optimal number of hash functions
    fn calculate_optimal_hashes(bits: usize, items: usize) -> usize {
        ((bits as f64 / items as f64) * std::f64::consts::LN_2).ceil() as usize
    }
    
    /// Add a value to the bloom filter
    pub fn add(&mut self, value: &Value) -> ChigutiroResult<()> {
        let hashes = self.hash_value(value);
        
        for hash in hashes {
            let bit_index = (hash % self.num_bits as u64) as usize;
            let byte_index = bit_index / 8;
            let bit_offset = bit_index % 8;
            
            if byte_index < self.bits.len() {
                self.bits[byte_index] |= 1 << bit_offset;
            }
        }
        
        self.items_added += 1;
        Ok(())
    }
    
    /// Check if a value might be in the set
    pub fn might_contain(&self, value: &Value) -> bool {
        let hashes = self.hash_value(value);
        
        for hash in hashes {
            let bit_index = (hash % self.num_bits as u64) as usize;
            let byte_index = bit_index / 8;
            let bit_offset = bit_index % 8;
            
            if byte_index >= self.bits.len() {
                return false;
            }
            
            if (self.bits[byte_index] & (1 << bit_offset)) == 0 {
                return false; // Definitely not in set
            }
        }
        
        true // Might be in set
    }
    
    /// Generate hash values for a value
    fn hash_value(&self, value: &Value) -> Vec<u64> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash the JSON value
        match value {
            Value::String(s) => s.hash(&mut hasher),
            Value::Number(n) => n.to_string().hash(&mut hasher),
            Value::Bool(b) => b.hash(&mut hasher),
            _ => value.to_string().hash(&mut hasher),
        }
        
        let hash1 = hasher.finish();
        
        // Generate second hash
        let mut hasher2 = DefaultHasher::new();
        hash1.hash(&mut hasher2);
        let hash2 = hasher2.finish();
        
        // Generate k hashes using double hashing
        (0..self.num_hashes)
            .map(|i| hash1.wrapping_add((i as u64).wrapping_mul(hash2)))
            .collect()
    }
    
    /// Get current false positive rate estimate
    pub fn estimated_false_positive_rate(&self) -> f64 {
        if self.items_added == 0 {
            return 0.0;
        }
        
        let bits_set = self.bits.iter()
            .map(|byte| byte.count_ones() as usize)
            .sum::<usize>();
        
        let proportion_set = bits_set as f64 / self.num_bits as f64;
        proportion_set.powi(self.num_hashes as i32)
    }
    
    /// Get index statistics  
    pub fn stats(&self) -> BloomIndexStats {
        let bits_set = self.bits.iter()
            .map(|byte| byte.count_ones() as usize)
            .sum::<usize>();
        
        BloomIndexStats {
            num_bits: self.num_bits,
            num_hashes: self.num_hashes,
            items_added: self.items_added,
            bits_set,
            fill_ratio: bits_set as f64 / self.num_bits as f64,
            estimated_fpr: self.estimated_false_positive_rate(),
            expected_fpr: self.false_positive_rate,
            memory_usage_bytes: self.bits.len(),
        }
    }
}

/// General index statistics
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub unique_values: usize,
    pub total_entries: usize,
    pub memory_usage_estimate: usize,
    pub optimized: bool,
}

/// Bloom filter specific statistics
#[derive(Debug, Clone)]
pub struct BloomIndexStats {
    pub num_bits: usize,
    pub num_hashes: usize,
    pub items_added: usize,
    pub bits_set: usize,
    pub fill_ratio: f64,
    pub estimated_fpr: f64,
    pub expected_fpr: f64,
    pub memory_usage_bytes: usize,
}

/// Compressed index for memory efficiency
pub struct CompressedIndex {
    /// Run-length encoded values
    compressed_data: Vec<u8>,
    
    /// Dictionary for value compression
    value_dictionary: HashMap<Value, u32>,
    
    /// Reverse dictionary
    reverse_dictionary: Vec<Value>,
    
    /// Index into compressed data
    index: Vec<usize>,
}

impl CompressedIndex {
    /// Create a new compressed index
    pub fn new() -> Self {
        Self {
            compressed_data: Vec::new(),
            value_dictionary: HashMap::new(),
            reverse_dictionary: Vec::new(),
            index: Vec::new(),
        }
    }
    
    /// Add a record to the compressed index
    pub fn add(&mut self, record_id: u64, value: &Value) -> ChigutiroResult<()> {
        // Get or create dictionary entry
        let value_id = if let Some(id) = self.value_dictionary.get(value) {
            *id
        } else {
            let id = self.reverse_dictionary.len() as u32;
            self.value_dictionary.insert(value.clone(), id);
            self.reverse_dictionary.push(value.clone());
            id
        };
        
        // Add to compressed data (simplified RLE)
        self.compressed_data.extend_from_slice(&record_id.to_le_bytes());
        self.compressed_data.extend_from_slice(&value_id.to_le_bytes());
        
        Ok(())
    }
    
    /// Get records for a value
    pub fn get_records(&self, value: &Value) -> ChigutiroResult<Vec<u64>> {
        if let Some(value_id) = self.value_dictionary.get(value) {
            let mut records = Vec::new();
            
            // Scan compressed data (simplified)
            let mut i = 0;
            while i + 11 < self.compressed_data.len() {
                let record_id = u64::from_le_bytes([
                    self.compressed_data[i], self.compressed_data[i+1],
                    self.compressed_data[i+2], self.compressed_data[i+3],
                    self.compressed_data[i+4], self.compressed_data[i+5],
                    self.compressed_data[i+6], self.compressed_data[i+7],
                ]);
                
                let stored_value_id = u32::from_le_bytes([
                    self.compressed_data[i+8], self.compressed_data[i+9],
                    self.compressed_data[i+10], self.compressed_data[i+11],
                ]);
                
                if stored_value_id == *value_id {
                    records.push(record_id);
                }
                
                i += 12;
            }
            
            Ok(records)
        } else {
            Ok(Vec::new())
        }
    }
    
    /// Get compression statistics
    pub fn compression_stats(&self) -> CompressionStats {
        let uncompressed_size = self.reverse_dictionary.len() * 
                               (std::mem::size_of::<Value>() + std::mem::size_of::<u64>());
        
        CompressionStats {
            dictionary_size: self.reverse_dictionary.len(),
            compressed_size: self.compressed_data.len(),
            uncompressed_estimate: uncompressed_size,
            compression_ratio: uncompressed_size as f64 / self.compressed_data.len() as f64,
        }
    }
}

/// Compression statistics
#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub dictionary_size: usize,
    pub compressed_size: usize,
    pub uncompressed_estimate: usize,
    pub compression_ratio: f64,
} 