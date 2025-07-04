#!/bin/bash

# Caminho para o diretório onde buscar os arquivos
BASE_DIR="/Users/cilasmfm/workspace/landsat-results/outputs/serial"

# Encontrar todos os arquivos time.csv e verificar se estão vazios

# Encontrar todos os arquivos time.csv em subdiretórios
find "$BASE_DIR" -type f -name "time.csv" | while read -r file; do
    # Conta o número de linhas no arquivo
    line_count=$(wc -l < "$file")

    # Verifica se o arquivo tem 3 linhas ou menos
    if [ "$line_count" -le 5 ]; then
        # Obtém o diretório do arquivo
        dir=$(dirname "$file")
        
        # Exclui o diretório e todo o seu conteúdo
        echo "Removendo diretório: $dir"
        rm -rf "$dir"
    fi
done