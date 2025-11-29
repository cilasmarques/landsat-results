#!/bin/bash

# Caminho base para os diretórios principais
BASE_DIRS=("/Users/cilasmfm/workspace/landsat-results/results/")

# Loop por cada diretório base
for base_dir in "${BASE_DIRS[@]}"; do
    # Verifica se o diretório base existe
    if [ -d "$base_dir" ]; then
        echo "Diretório principal: $base_dir"
        
        # Encontrar todos os subdiretórios de primeiro nível
        find "$base_dir" -mindepth 1 -maxdepth 1 -type d | while read -r sub_dir; do
            # Conta o número de subdiretórios dentro de cada subdiretório encontrado
            subdir_count=$(find "$sub_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
            
            # Exibe o resultado para cada subdiretório
            echo "  Subdiretório: $(basename "$sub_dir") - Subdiretórios internos: $subdir_count"
        done
        echo
    else
        echo "Diretório $base_dir não encontrado"
    fi
done
