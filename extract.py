import os
import re

def extract_files(md_file_path):
    with open(md_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all ## sections
    sections = re.split(r'^##\s+', content, flags=re.MULTILINE)
    
    for section in sections[1:]:
        lines = section.split('\n')
        header = lines[0].strip()
        
        # Remove emojis and leading/trailing spaces
        header = re.sub(r'[^\w\.\-\/\s]', '', header).strip()
        
        # The filename is usually the first word in the header before '—' or space
        filename = header.split('—')[0].strip()
        if ' ' in filename:
            filename = filename.split()[0].strip()
            
        if not filename or filename == 'Project' or filename == 'SeniorLevel':
            continue
            
        # Extract the code block
        code_block_match = re.search(r'```[a-z]*\n(.*?)```', section, flags=re.DOTALL)
        if code_block_match:
            file_content = code_block_match.group(1)
            
            # Create directories
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8', newline='\n') as f:
                f.write(file_content)
            try:
                print(f"Extracted: {filename}")
            except Exception:
                pass

if __name__ == '__main__':
    extract_files('skills/recurmint (1).md')
