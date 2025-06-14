import re
from typing import Dict, List, Any, Optional, Union, Tuple
import math


class ChunkProcessor:
    """Process and chunk text and code for analysis or embedding."""
    
    def __init__(self, max_chunk_size: int = 1024, overlap: int = 128):
        """Initialize the chunk processor.
        
        Args:
            max_chunk_size: Maximum size of each chunk in characters.
            overlap: Number of characters to overlap between chunks.
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text into overlapping segments.
        
        Args:
            text: The text to chunk.
            
        Returns:
            A list of text chunks.
        """
        chunks = []
        
        # If text is shorter than max chunk size, return it as a single chunk
        if len(text) <= self.max_chunk_size:
            return [text]
        
        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        current_size = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            paragraph_size = len(paragraph)
            
            # If paragraph is larger than max_chunk_size, split it into sentences
            if paragraph_size > self.max_chunk_size:
                # Add current chunk if not empty
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                    current_size = 0
                
                # Split large paragraph into sentences
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                sentence_chunks = self._chunk_sentences(sentences)
                chunks.extend(sentence_chunks)
            elif current_size + paragraph_size + 1 <= self.max_chunk_size:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                    current_size += paragraph_size + 2
                else:
                    current_chunk = paragraph
                    current_size = paragraph_size
            else:
                # Start a new chunk
                chunks.append(current_chunk)
                current_chunk = paragraph
                current_size = paragraph_size
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        # Add overlapping content
        if self.overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)
        
        return chunks
    
    def _chunk_sentences(self, sentences: List[str]) -> List[str]:
        """Chunk a list of sentences.
        
        Args:
            sentences: List of sentences to chunk.
            
        Returns:
            A list of chunks, each containing one or more sentences.
        """
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_size = len(sentence)
            
            # If sentence is larger than max_chunk_size, split it
            if sentence_size > self.max_chunk_size:
                # Add current chunk if not empty
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                    current_size = 0
                
                # Split the long sentence
                start = 0
                while start < sentence_size:
                    end = min(start + self.max_chunk_size, sentence_size)
                    chunks.append(sentence[start:end])
                    start = end
            elif current_size + sentence_size + 1 <= self.max_chunk_size:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                    current_size += sentence_size + 1
                else:
                    current_chunk = sentence
                    current_size = sentence_size
            else:
                # Start a new chunk
                chunks.append(current_chunk)
                current_chunk = sentence
                current_size = sentence_size
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlapping content between chunks.
        
        Args:
            chunks: List of chunks.
            
        Returns:
            A list of chunks with overlapping content.
        """
        if not chunks or len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i in range(len(chunks)):
            chunk = chunks[i]
            
            # Add overlap from the previous chunk
            if i > 0:
                prev_chunk = chunks[i-1]
                overlap_text = prev_chunk[-min(self.overlap, len(prev_chunk)):]
                chunk = f"[CONTEXT_OVERLAP: {overlap_text}]\n\n{chunk}"
            
            # Add overlap from the next chunk
            if i < len(chunks) - 1:
                next_chunk = chunks[i+1]
                overlap_text = next_chunk[:min(self.overlap, len(next_chunk))]
                chunk = f"{chunk}\n\n[CONTINUES: {overlap_text}]"
            
            overlapped_chunks.append(chunk)
        
        return overlapped_chunks
    
    def chunk_code(self, code: str) -> List[Dict[str, Any]]:
        """Chunk code into logical segments with metadata.
        
        Args:
            code: The code to chunk.
            
        Returns:
            A list of dictionaries with code chunks and metadata.
        """
        chunks = []
        
        # Check if we need to chunk at all
        if len(code) <= self.max_chunk_size:
            return [{
                'content': code,
                'start_line': 1,
                'end_line': code.count('\n') + 1,
                'type': 'full_file'
            }]
        
        # Split into lines for better chunking
        lines = code.split('\n')
        num_lines = len(lines)
        
        # Try to extract functions/blocks
        blocks = self._extract_code_blocks(code, lines)
        
        if blocks:
            # Start with any code before the first block
            first_block_start = blocks[0]['start_line']
            if first_block_start > 1:
                preamble = '\n'.join(lines[:first_block_start-1])
                chunks.append({
                    'content': preamble,
                    'start_line': 1,
                    'end_line': first_block_start - 1,
                    'type': 'preamble'
                })
            
            # Add each block
            for i, block in enumerate(blocks):
                # If block is too large, split it
                block_lines = block['end_line'] - block['start_line'] + 1
                block_content = '\n'.join(lines[block['start_line']-1:block['end_line']])
                
                if len(block_content) > self.max_chunk_size:
                    # Split large block
                    sub_chunks = self._split_large_block(block_content, block['start_line'])
                    chunks.extend(sub_chunks)
                else:
                    # Add the block as is
                    chunks.append({
                        'content': block_content,
                        'start_line': block['start_line'],
                        'end_line': block['end_line'],
                        'type': block['type'],
                        'name': block.get('name')
                    })
                
                # Add any code between this block and the next
                if i < len(blocks) - 1:
                    next_block_start = blocks[i+1]['start_line']
                    if next_block_start > block['end_line'] + 1:
                        interstitial = '\n'.join(lines[block['end_line']:next_block_start-1])
                        chunks.append({
                            'content': interstitial,
                            'start_line': block['end_line'] + 1,
                            'end_line': next_block_start - 1,
                            'type': 'interstitial'
                        })
            
            # Add any code after the last block
            last_block_end = blocks[-1]['end_line']
            if last_block_end < num_lines:
                postamble = '\n'.join(lines[last_block_end:])
                chunks.append({
                    'content': postamble,
                    'start_line': last_block_end + 1,
                    'end_line': num_lines,
                    'type': 'postamble'
                })
        else:
            # No blocks found, chunk by size
            chunks = self._chunk_by_size(code, lines)
        
        return chunks
    
    def _extract_code_blocks(self, code: str, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract logical code blocks from the code.
        
        Args:
            code: The code.
            lines: The code split into lines.
            
        Returns:
            A list of dictionaries with block metadata.
        """
        blocks = []
        num_lines = len(lines)
        
        # Look for function declarations
        js_function_pattern = r'(?:function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)|(?:const|let|var)?\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*(?:function|\([^)]*\)\s*=>))'
        
        # Track opening and closing braces to find block boundaries
        brace_stack = []
        current_function_start = None
        current_function_name = None
        
        for i, line in enumerate(lines):
            # Look for function declarations
            if current_function_start is None:
                match = re.search(js_function_pattern, line)
                if match:
                    current_function_start = i + 1
                    current_function_name = match.group(1) if match.group(1) else match.group(2)
            
            # Count braces
            for char in line:
                if char == '{':
                    brace_stack.append(i + 1)
                elif char == '}' and brace_stack:
                    open_brace_line = brace_stack.pop()
                    
                    # Check if this closes a function
                    if current_function_start is not None and open_brace_line >= current_function_start and not brace_stack:
                        blocks.append({
                            'start_line': current_function_start,
                            'end_line': i + 1,
                            'type': 'function',
                            'name': current_function_name
                        })
                        current_function_start = None
                        current_function_name = None
        
        # Also look for class declarations
        class_pattern = r'class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)'
        class_matches = re.finditer(class_pattern, code)
        
        for match in class_matches:
            class_name = match.group(1)
            class_start_line = code[:match.start()].count('\n') + 1
            
            # Find the end of the class by matching braces
            brace_count = 0
            class_end_line = class_start_line
            
            for i in range(class_start_line - 1, num_lines):
                line = lines[i]
                for char in line:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            class_end_line = i + 1
                            break
                
                if brace_count == 0 and class_end_line > class_start_line:
                    break
            
            if class_end_line > class_start_line:
                blocks.append({
                    'start_line': class_start_line,
                    'end_line': class_end_line,
                    'type': 'class',
                    'name': class_name
                })
        
        # Sort blocks by start line
        blocks.sort(key=lambda x: x['start_line'])
        
        return blocks
    
    def _split_large_block(self, block_content: str, start_line: int) -> List[Dict[str, Any]]:
        """Split a large code block into smaller chunks.
        
        Args:
            block_content: The content of the block.
            start_line: The starting line number of the block.
            
        Returns:
            A list of dictionaries with block chunks.
        """
        chunks = []
        lines = block_content.split('\n')
        num_lines = len(lines)
        
        # Estimate how many lines we can fit in a chunk
        avg_line_length = len(block_content) / num_lines
        lines_per_chunk = min(num_lines, math.floor(self.max_chunk_size / avg_line_length))
        
        # Create chunks
        for i in range(0, num_lines, lines_per_chunk):
            end_idx = min(i + lines_per_chunk, num_lines)
            chunk_content = '\n'.join(lines[i:end_idx])
            
            chunks.append({
                'content': chunk_content,
                'start_line': start_line + i,
                'end_line': start_line + end_idx - 1,
                'type': 'block_chunk'
            })
        
        return chunks
    
    def _chunk_by_size(self, code: str, lines: List[str]) -> List[Dict[str, Any]]:
        """Chunk code by size when no logical blocks are found.
        
        Args:
            code: The code.
            lines: The code split into lines.
            
        Returns:
            A list of dictionaries with code chunks.
        """
        chunks = []
        num_lines = len(lines)
        
        # Estimate how many lines we can fit in a chunk
        avg_line_length = len(code) / num_lines
        lines_per_chunk = min(num_lines, math.floor(self.max_chunk_size / avg_line_length))
        
        # Create chunks
        for i in range(0, num_lines, lines_per_chunk):
            end_idx = min(i + lines_per_chunk, num_lines)
            chunk_content = '\n'.join(lines[i:end_idx])
            
            chunks.append({
                'content': chunk_content,
                'start_line': i + 1,
                'end_line': end_idx,
                'type': 'chunk'
            })
        
        return chunks
    
    def combine_chunks(self, chunks: List[Dict[str, Any]], max_context_size: int) -> List[Dict[str, Any]]:
        """Combine chunks to fit within a maximum context size.
        
        This is useful for creating chunks that fit within an LLM context window.
        
        Args:
            chunks: List of chunk dictionaries.
            max_context_size: Maximum combined size.
            
        Returns:
            A list of combined chunk dictionaries.
        """
        combined_chunks = []
        current_chunk = None
        current_size = 0
        
        for chunk in chunks:
            chunk_size = len(chunk['content'])
            
            if current_chunk is None:
                # Start a new combined chunk
                current_chunk = {
                    'content': chunk['content'],
                    'chunks': [chunk],
                    'start_line': chunk['start_line'],
                    'end_line': chunk['end_line']
                }
                current_size = chunk_size
            elif current_size + chunk_size <= max_context_size:
                # Add to current combined chunk
                current_chunk['content'] += '\n\n' + chunk['content']
                current_chunk['chunks'].append(chunk)
                current_chunk['end_line'] = chunk['end_line']
                current_size += chunk_size + 2  # +2 for newlines
            else:
                # Start a new combined chunk
                combined_chunks.append(current_chunk)
                current_chunk = {
                    'content': chunk['content'],
                    'chunks': [chunk],
                    'start_line': chunk['start_line'],
                    'end_line': chunk['end_line']
                }
                current_size = chunk_size
        
        # Add the last combined chunk
        if current_chunk:
            combined_chunks.append(current_chunk)
        
        return combined_chunks 