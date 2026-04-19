#!/usr/bin/env python3
"""
LLVM IR Normalizer for Machine Learning
Normalizes LLVM IR code by:
1. Removing function names from define statements
2. Normalizing constants
3. Keeping declare statements unchanged
4. Using LLVMLite for accurate parsing
"""

import sys
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict
import re
import os
import contextlib
import typer


@contextlib.contextmanager
def suppress_stderr():
    """
    一个上下文管理器，用于临时抑制对 stderr 的写入。
    它通过在操作系统级别重定向文件描述符来实现，因此对 C 扩展也有效。
    """
    # os.devnull 是一个跨平台的空设备路径 (在 Unix 上是 '/dev/null')
    devnull = open(os.devnull, 'w')
    # 获取 stderr 的原始文件描述符 (通常是 2)
    stderr_fd = sys.stderr.fileno()
    # 复制一份原始的 stderr 文件描述符，以便之后恢复
    saved_stderr_fd = os.dup(stderr_fd)

    try:
        # 将 stderr 的文件描述符重定向到空设备
        os.dup2(devnull.fileno(), stderr_fd)
        
        # yield 关键字是上下文管理器的核心，您的代码将在这里执行
        yield
    finally:
        # 无论 try 块中发生什么（即使是异常），finally 块都会执行
        
        # 将 stderr 恢复到原始状态
        os.dup2(saved_stderr_fd, stderr_fd)
        # 关闭我们保存的描述符副本
        os.close(saved_stderr_fd)
        # 关闭我们打开的空设备文件
        devnull.close()
        

try:
    import llvmlite.ir as ll
    import llvmlite.binding as llvm
    LLVMLITE_AVAILABLE = True
except ImportError:
    LLVMLITE_AVAILABLE = False
    print("Warning: LLVMLite not available, falling back to regex-based parsing")
    print("Install with: pip install llvmlite")


class LLVMIRNormalizer:
    def __init__(self):
        # Counter for normalized identifiers
        self.var_counter = 0
        self.bb_counter = 0
        self.func_counter = 0
        
        # Maps for consistent renaming
        self.var_map = {}
        self.bb_map = {}
        self.func_map = {}
        
        # Fallback regex patterns for when LLVMLite is not available
        self.patterns = {
            'define': re.compile(r'^define\s+.*?\s+@(\w+)\s*\(', re.MULTILINE),
            'declare': re.compile(r'^declare\s+.*?\s+@(\w+)\s*\(', re.MULTILINE),
            'variable': re.compile(r'%([.\w]+)'),
            'basic_block': re.compile(r'^(\w+):'),
            'constant_int': re.compile(r'\b(\d+)\b'),
            'constant_float': re.compile(r'\b(\d+\.\d+)\b'),
            'constant_hex': re.compile(r'\b0x[0-9a-fA-F]+\b'),
            'function_call': re.compile(r'@(\w+)\s*\('),
            'global_var': re.compile(r'@(\w+)'),
            # Addtional patterns for metadata and other constructs
            'regular_function_names': re.compile(r'@(\w[\w\d\.]*)'),
            'quoted_function_names': re.compile(r'@"([^"]*)"'),
            # Normalize Constants
            'bb_label_def': re.compile(r'^([^:]+):\s*$'),
            'bb_ref': re.compile(r'(label\s+%"[^"]*"|%"[^"]*"|\s@"[^"]*")'),
            'function_ref': re.compile(r'@[^\s,\)\]:]+'),
            'line_interger_constants': re.compile(r'(?<!i)(?<!\[)(?<!%\.)(\b\d+\b)(?![*.]|\s*x\s)'),
            'line_float_constants': re.compile(r'(?<!%\.)\b\d+\.\d+(?:e[+-]?\d+)?\b')
        }
    
    def normalize_with_llvmlite(self, ir_code: str) -> str:
        """Normalize LLVM IR using LLVMLite for accurate parsing"""
        if not LLVMLITE_AVAILABLE:
            return self.normalize_with_regex(ir_code)
        
        try:
            # Initialize LLVM
            llvm.initialize()
            llvm.initialize_native_target()
            llvm.initialize_native_asmprinter()
            
            # Parse the IR
            with suppress_stderr():
                mod = llvm.parse_assembly(ir_code)
            
            # Convert to string and normalize
            ir_str = str(mod)
            normalized_ir = self._normalize_parsed_ir(ir_str)
            
            return normalized_ir
            
        except Exception as e:
            return self.normalize_with_regex(ir_code)
    
    def _normalize_parsed_ir(self, ir_str: str) -> str:
        """Normalize parsed LLVM IR string"""
        # Remove metadata definitions at the end of the file
        ir_str = self._remove_metadata_definitions(ir_str)
        
        lines = ir_str.split('\n')
        normalized_lines = []
        
        for line in lines:
            normalized_line = self._normalize_llvmlite_line(line)
            normalized_lines.append(normalized_line)
        
        return '\n'.join(normalized_lines).strip()
    
    def _normalize_llvmlite_line(self, line: str) -> str:
        """Normalize a single line from LLVMLite parsed IR"""
        original_line = line.strip()
        
        # Skip empty lines and comments
        if not original_line or original_line.startswith(';'):
            return line
        
        # Handle define statements - remove function name but keep signature
        if original_line.startswith('define'):
            # Use regex to extract and replace function name
            match = re.search(r'@(\w+)\s*\(', line)
            if match:
                func_name = match.group(1)
                normalized_func = self.get_normalized_func(func_name)
                line = line.replace(f'@{func_name}', f'@{normalized_func}')
        
        # Handle declare statements - keep unchanged for external functions
        elif original_line.startswith('declare'):
            # Don't normalize declare statements as requested
            return line
        
        # Normalize variables
        line = self._normalize_variables(line)
        
        # Normalize basic blocks
        line = self._normalize_basic_blocks(line)
        
        # Normalize constants
        line = self._normalize_constants_advanced(line)
        
        # Normalize function calls
        line = self._normalize_function_calls(line)
        
        return line
    
    def _normalize_variables(self, line: str) -> str:
        """Normalize LLVM variables (%var)"""
        def replace_var(match):
            var_name = match.group(1)
            # Skip numeric SSA form variables
            if var_name.isdigit():
                return match.group(0)
            # Skip variables that start with dot (like %.160)
            if var_name.startswith('.'):
                return match.group(0)
            return f'%{self.get_normalized_var(var_name)}'
        
        return re.sub(r'%([.\w]+)', replace_var, line)
    
    def _normalize_basic_blocks(self, line: str) -> str:
        """Normalize basic block labels - keep original labels for basic blocks"""
        # Basic block definition (e.g., "bb1:" or "@bb1:" or "\"@bb1\":")
        bb_match = re.match(r'^"?([^":]+)"?:\s*$', line.strip())
        if bb_match:
            # Keep original basic block labels - don't normalize them
            return line
        
        # Basic block references in branch instructions - keep original
        # No need to normalize basic block references in branch instructions
        return line
    
    def _normalize_constants_advanced(self, line: str) -> str:
        """Advanced constant normalization"""
        # First, protect metadata references, basic block labels, and basic block references
        metadata_refs = []
        bb_refs = []
        bb_label_refs = []
        
        def protect_metadata(match):
            metadata_refs.append(match.group(0))
            return f'__METADATA_{len(metadata_refs)-1}__'
        
        def protect_bb_labels(match):
            bb_refs.append(match.group(0))
            return f'__BB_LABEL_{len(bb_refs)-1}__'
            
        def protect_bb_label_refs(match):
            bb_label_refs.append(match.group(0))
            return f'__BB_LABEL_REF_{len(bb_label_refs)-1}__'
        
        # Protect metadata references
        metadata_pattern = r'!\d+'
        line = re.sub(metadata_pattern, protect_metadata, line)
        
        # Protect basic block label definitions (lines ending with colon)
        # Examples: "@0":, "@2":, "@3":, "bb1":, etc.
        bb_label_def_pattern = r'^([^:]+):\s*$'
        if re.match(bb_label_def_pattern, line.strip()):
            # This is a basic block label definition line, don't modify it
            return line
        
        # Protect basic block references in branch instructions
        # Examples: label %"@3", label %"@2", %"@0", etc.
        bb_ref_pattern = r'(label\s+%"[^"]*"|%"[^"]*"|\s@"[^"]*")'
        line = re.sub(bb_ref_pattern, protect_bb_label_refs, line)
        
        # Protect function references and other @ symbols
        # Examples: @_Z9etoupperww, @towupper, etc.
        function_ref_pattern = r'@[^\s,\)\]:]+'
        line = re.sub(function_ref_pattern, protect_bb_labels, line)
        
        # Integer constants (preserve common small values and alignment values)
        def replace_int(match):
            try:
                value = int(match.group(1))
                # Preserve very common small values and alignment values
                if value in [0, 1, -1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64, 73, 105, 128, 256]:
                    return str(value)
                # Preserve small numbers that often have semantic meaning
                elif abs(value) <= 16:
                    return str(value)
                elif abs(value) < 100:
                    return "<MED_INT>"
                elif abs(value) < 1000:
                    return "<LARGE_INT>"
                else:
                    return "<HUGE_INT>"
            except:
                return match.group(0)
        
        # Hexadecimal constants
        line = re.sub(r'\b0x[0-9a-fA-F]+\b', '<HEX_CONST>', line)
        
        # Integer constants (avoid matching array sizes in type declarations and register names)
        # Avoid matching array sizes in type declarations like [12 x i8]
        # Avoid matching register names like %.160
        line = re.sub(r'(?<!i)(?<!\[)(?<!%\.)(\b\d+\b)(?![*.]|\s*x\s)', replace_int, line)
        
        # Float constants (avoid matching register names like %.160)
        line = re.sub(r'(?<!%\.)\b\d+\.\d+(?:e[+-]?\d+)?\b', '<FLOAT_CONST>', line)
        
        # String constants - but be very careful not to match basic block labels
        # Only match actual string literals in double quotes that are not basic block references
        string_pattern = r'"[^"@%]*"(?!\s*:)'  # Don't match if followed by colon (basic block label)
        line = re.sub(string_pattern, '<STRING_CONST>', line)
        
        # Restore basic block label references first
        for i, bb_label_ref in enumerate(bb_label_refs):
            line = line.replace(f'__BB_LABEL_REF_{i}__', bb_label_ref)
        
        # Restore basic block labels
        for i, bb_ref in enumerate(bb_refs):
            line = line.replace(f'__BB_LABEL_{i}__', bb_ref)
        
        # Restore metadata references
        for i, metadata_ref in enumerate(metadata_refs):
            line = line.replace(f'__METADATA_{i}__', metadata_ref)
        
        return line
    
    def _normalize_function_calls(self, line: str) -> str:
        """Normalize function calls and references"""
        # Function calls (@func) - handle both regular and quoted names
        def replace_func_call(match):
            func_name = match.group(1)
            # Skip already normalized functions
            if func_name.startswith('func'):
                return match.group(0)
            # Skip already normalized basic blocks
            if func_name.startswith('bb'):
                return match.group(0)
            # Skip basic block references (numeric or starting with digit) - keep original
            # Examples: @0, @1, @2.i2, @4.loopexit, etc.
            if func_name.isdigit() or (func_name and func_name[0].isdigit()):
                return match.group(0)  # Keep original basic block reference
            # Skip intrinsic functions (like llvm.*)
            if func_name.startswith('llvm.'):
                return match.group(0)
            # Skip global variables that start with dot
            if func_name.startswith('.'):
                return match.group(0)
            return f'@{self.get_normalized_func(func_name)}'
        
        def replace_quoted_func_call(match):
            func_name = match.group(1)
            # Skip already normalized functions
            if func_name.startswith('func'):
                return match.group(0)
            # Skip already normalized basic blocks
            if func_name.startswith('bb'):
                return match.group(0)
            # Skip basic block references (numeric or starting with digit) - keep original
            # Examples: @"0", @"1", @"2.i2", @"4.loopexit", etc.
            if func_name.isdigit() or (func_name and func_name[0].isdigit()):
                return match.group(0)  # Keep original basic block reference
            # Skip intrinsic functions (like llvm.*)
            if func_name.startswith('llvm.'):
                return match.group(0)
            # Skip global variables that start with dot
            if func_name.startswith('.'):
                return match.group(0)
            return f'@{self.get_normalized_func(func_name)}'
        
        # Handle regular function names
        line = self.patterns['regular_function_names'].sub(replace_func_call, line)
        # Handle quoted function names
        line = self.patterns['quoted_function_names'].sub(replace_quoted_func_call, line)

        return line
    
    def normalize_with_regex(self, ir_code: str) -> str:
        """Fallback regex-based normalization"""
        
        # Remove 
        # source_filename = "./IR/BinaryCorp/small_train/small_train/2bwm-git-2bwm-O0-f9579e061f6e200bc50fdae0d8f2a873.ll"
        # target datalayout = "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-i128:128-f64:32:64-f80:32-n8:16:32-S128"
        # target triple = "i386-pc-linux-gnu"

        ir_code = ir_code.strip()
        # Remove source file name
        for line in ir_code.splitlines():
            if line.startswith("source_filename =") or line.startswith("target datalayout =") or line.startswith("target triple ="):
                ir_code = ir_code.replace(line, "")

        # Remove metadata definitions at the end of the file
        ir_code = self._remove_metadata_definitions(ir_code)

        # Remove comments
        ir_code = '\n'.join([line for line in ir_code.splitlines() if not line.strip().startswith(';')])

        # Remove comments at the end of lines
        ir_code = '\n'.join([line.split(';')[0].strip() for line in ir_code.splitlines()])
        
        lines = ir_code.split('\n')
        normalized_lines = []
        
        for line in lines:
            normalized_line = self.normalize_line(line)
            normalized_lines.append(normalized_line)
        
        return '\n'.join(normalized_lines).strip()
    
    def _remove_metadata_definitions(self, ir_code: str) -> str:
        """Remove metadata definitions at the end of the file (e.g., !0 = !{i32 0})"""
        lines = ir_code.split('\n')
        
        # Find the last line that is not a metadata definition
        last_content_line = -1
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            # Skip empty lines and comments
            if not stripped_line or stripped_line.startswith(';'):
                continue
            # If this line is a metadata definition, continue looking
            if re.match(r'^!\d+\s*=\s*!?\{.*\}$', stripped_line):
                continue
            # This is actual content, record it
            last_content_line = i
        
        # If we found content, keep everything up to and including the last content line
        if last_content_line >= 0:
            # Keep all lines up to last_content_line, and filter out metadata definitions after
            result_lines = lines[:last_content_line + 1]
            
            # Add any remaining non-metadata lines after the last content
            for i in range(last_content_line + 1, len(lines)):
                stripped_line = lines[i].strip()
                # Keep non-metadata lines (like attributes, empty lines, etc.)
                if (not stripped_line or 
                    stripped_line.startswith(';') or 
                    stripped_line.startswith('attributes') or
                    not re.match(r'^!\d+\s*=\s*!?\{.*\}$', stripped_line)):
                    result_lines.append(lines[i])
            
            return '\n'.join(result_lines)
        
        # If no content found, return as is
        return ir_code
    
    def normalize_constants(self, line: str) -> str:
        """Normalize constants in a line"""
        # First, protect metadata references, basic block labels, and basic block references
        metadata_refs = []
        bb_refs = []
        bb_label_refs = []
        
        def protect_metadata(match):
            metadata_refs.append(match.group(0))
            return f'__METADATA_{len(metadata_refs)-1}__'
        
        def protect_bb_labels(match):
            bb_refs.append(match.group(0))
            return f'__BB_LABEL_{len(bb_refs)-1}__'
            
        def protect_bb_label_refs(match):
            bb_label_refs.append(match.group(0))
            return f'__BB_LABEL_REF_{len(bb_label_refs)-1}__'
        
        # Protect metadata references
        metadata_pattern = r'!\d+'
        line = re.sub(metadata_pattern, protect_metadata, line)
        
        # Protect basic block label definitions (lines ending with colon)
        # Examples: "@0":, "@2":, "@3":, "bb1":, etc.
        if self.patterns['bb_label_def'].match(line.strip()):
            # This is a basic block label definition line, don't modify it
            return line
        
        # Protect basic block references in branch instructions
        # Examples: label %"@3", label %"@2", %"@0", etc.
        line = self.patterns['bb_ref'].sub(protect_bb_label_refs, line)
        
        # Protect function references and other @ symbols
        # Examples: @_Z9etoupperww, @towupper, etc.
        line = self.patterns['function_ref'].sub(protect_bb_labels, line)
        
        # Normalize integer constants (preserve common small values and alignment values)
        def replace_int(match):
            value = int(match.group(1))
            # Preserve very common small values and alignment values
            if value in [0, 1, -1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64, 128, 256]:
                return str(value)
            # Preserve small numbers that often have semantic meaning
            elif abs(value) <= 16:
                return str(value)
            elif abs(value) < 100:
                return "<MED_INT>"
            elif abs(value) < 1000:
                return "<LARGE_INT>"
            else: 
                return "<HUGE_INT>"
        
        # Normalize hexadecimal constants
        line = self.patterns['constant_hex'].sub('<HEX_CONST>', line)
        
        # Normalize integer constants (avoid matching array sizes in type declarations and register names)
        # Avoid matching register names like %.160
        line = self.patterns['line_interger_constants'].sub(replace_int, line)
        
        # Normalize float constants (avoid matching register names like %.160)
        line = self.patterns['line_float_constants'].sub('<FLOAT_CONST>', line)
        
        # Restore basic block label references first
        for i, bb_label_ref in enumerate(bb_label_refs):
            line = line.replace(f'__BB_LABEL_REF_{i}__', bb_label_ref)
        
        # Restore basic block labels
        for i, bb_ref in enumerate(bb_refs):
            line = line.replace(f'__BB_LABEL_{i}__', bb_ref)
        
        # Restore metadata references
        for i, metadata_ref in enumerate(metadata_refs):
            line = line.replace(f'__METADATA_{i}__', metadata_ref)
        
        return line
    
    def get_normalized_var(self, var_name: str) -> str:
        """Get normalized variable name"""
        if var_name not in self.var_map:
            self.var_map[var_name] = f"var{self.var_counter}"
            self.var_counter += 1
        return self.var_map[var_name]
    
    def get_normalized_bb(self, bb_name: str) -> str:
        """Get normalized basic block name"""
        if bb_name not in self.bb_map:
            self.bb_map[bb_name] = f"bb{self.bb_counter}"
            self.bb_counter += 1
        return self.bb_map[bb_name]
    
    def get_normalized_func(self, func_name: str) -> str:
        """Get normalized function name"""
        if func_name not in self.func_map:
            self.func_map[func_name] = f"func{self.func_counter}"
            self.func_counter += 1
        return self.func_map[func_name]
    
    def normalize_line(self, line: str) -> str:
        """Normalize a single line of LLVM IR"""
        original_line = line.strip()
        
        # Skip empty lines and comments
        if not original_line or original_line.startswith(';'):
            return line
        
        # Handle define statements - remove function name
        if original_line.startswith('define'):
            # Extract function signature and replace function name
            match = self.patterns['define'].search(original_line)
            if match:
                func_name = match.group(1)
                normalized_func = self.get_normalized_func(func_name)
                line = line.replace(f'@{func_name}', f'@{normalized_func}')
        
        # Handle declare statements - keep as is but normalize function names for consistency
        elif original_line.startswith('declare'):
            match = self.patterns['declare'].search(original_line)
            if match:
                func_name = match.group(1)
                normalized_func = self.get_normalized_func(func_name)
                line = line.replace(f'@{func_name}', f'@{normalized_func}')
        
        # Handle basic block labels - keep original labels
        elif ':' in original_line and not any(op in original_line for op in ['=', 'call', 'invoke', 'load', 'store']):
            bb_match = self.patterns['basic_block'].match(original_line)
            if bb_match:
                # Keep original basic block labels - don't normalize them
                pass  # Keep the line as is
        
        # Normalize variables (%)
        def replace_var(match):
            var_name = match.group(1)
            # Skip numeric variables (they're already normalized)
            if var_name.isdigit():
                return match.group(0)
            # Skip variables that start with dot (like %.160)
            if var_name.startswith('.'):
                return match.group(0)
            return f'%{self.get_normalized_var(var_name)}'
        
        line = self.patterns['variable'].sub(replace_var, line)
        
        # Normalize function calls and global variables (@)
        def replace_global(match):
            name = match.group(1)
            # Skip if it's already normalized
            if name.startswith(('func', 'var', 'bb')):
                return match.group(0)
            # Skip basic block references (numeric or starting with digit) - keep original
            # Examples: @0, @1, @2.i2, @4.loopexit, etc.
            if name.isdigit() or (name and name[0].isdigit()):
                return match.group(0)  # Keep original basic block reference
            # Skip intrinsic functions
            if name.startswith('llvm.'):
                return match.group(0)
            # Skip global variables that start with dot
            if name.startswith('.'):
                return match.group(0)
            return f'@{self.get_normalized_func(name)}'
        
        def replace_quoted_global(match):
            name = match.group(1)
            # Skip if it's already normalized
            if name.startswith(('func', 'var', 'bb')):
                return match.group(0)
            # Skip basic block references (numeric or starting with digit) - keep original
            # Examples: @"0", @"1", @"2.i2", @"4.loopexit", etc.
            if name.isdigit() or (name and name[0].isdigit()):
                return match.group(0)  # Keep original basic block reference
            # Skip intrinsic functions
            if name.startswith('llvm.'):
                return match.group(0)
            # Skip global variables that start with dot
            if name.startswith('.'):
                return match.group(0)
            return f'@{self.get_normalized_func(name)}'
        
        # Handle regular function names
        line = self.patterns['regular_function_names'].sub(replace_global, line)
        # Handle quoted function names
        line = self.patterns['quoted_function_names'].sub(replace_quoted_global, line)

        # Normalize constants
        line = self.normalize_constants(line)
        
        return line
    
    def normalize_ir(self, ir_code: str) -> str:
        """Normalize entire LLVM IR code"""
        # Try LLVMLite first, fall back to regex if needed
        
        if LLVMLITE_AVAILABLE:
            return self.normalize_with_llvmlite(ir_code)
        else:
            return self.normalize_with_regex(ir_code)
    
    def reset(self):
        """Reset the normalizer state"""
        self.var_counter = 0
        self.bb_counter = 0
        self.func_counter = 0
        self.var_map.clear()
        self.bb_map.clear()
        self.func_map.clear()


def normalize_file(input_path: str, output_path: str = None) -> str:
    """Normalize an LLVM IR file"""
    normalizer = LLVMIRNormalizer()
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            ir_code = f.read()
        
        normalized_code = normalizer.normalize_ir(ir_code)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(normalized_code)
            print(f"Normalized IR saved to: {output_path}")
        
        return normalized_code
    
    except Exception as e:
        print(f"Error normalizing file {input_path}: {e}")
        return ""

def normalize_string(ir_code: str) -> str:
    """Normalize a string containing LLVM IR code"""
    normalizer = LLVMIRNormalizer()
    
    try:
        normalized_code = normalizer.normalize_ir(ir_code)
        return normalized_code
    except Exception as e:
        print(f"Error normalizing string: {e}")
        return ""


app = typer.Typer(add_completion=False)


@app.command()
def main(
    input_file: str = typer.Argument(..., help="Input LLVM IR file path"),
    output_file: Optional[str] = typer.Argument(None, help="Output file path"),
):
    """Normalize LLVM IR from a file."""
    result = normalize_file(input_file, output_file)
    if not output_file:
        print(result)


if __name__ == "__main__":
    app()