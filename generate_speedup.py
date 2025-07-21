#!/usr/bin/env python3
"""
Script para calcular o speedup por fase (P*) de cada estratégia em relação ao serial.
Gera dois CSVs: um para STEEP e um para SEBAL.
"""
import os
import csv
from collections import defaultdict

# Caminhos
BASE_DIR = 'summarized_results'
STEPP_SUFFIX = '-steep'
SEBAL_SUFFIX = '-sebal'
CSV_NAME = 'final-time.csv'

# Função para ler tempos medianos das fases P*_ de um arquivo
def ler_fases_p_medianos(path):
    fases = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['PHASE'] == 'PIXEL_FILTER':
                continue  # Ignorar PIXEL_FILTER
            if row['PHASE'].startswith('P') and ('_' in row['PHASE'] or row['PHASE'] == 'P_TOTAL' or row['PHASE'] == 'P5_COPY_HOST'):
                fases[row['PHASE']] = float(row['TIMESTAMP_median'])
    return fases

# Função para coletar todas as estratégias de um tipo (steep/sebal)
def coletar_estrategias(suffix):
    return sorted([d for d in os.listdir(BASE_DIR) if d.endswith(suffix)])

# Função para montar tabela de speedup
def gerar_speedup_csv(suffix, output_csv):
    estrategias = coletar_estrategias(suffix)
    if not estrategias:
        print(f'Nenhuma estratégia encontrada para {suffix}')
        return
    # Serial sempre existe
    serial_dir = [d for d in estrategias if d.startswith('serial')][0]
    serial_fases = ler_fases_p_medianos(os.path.join(BASE_DIR, serial_dir, CSV_NAME))
    # Coletar fases de cada estratégia
    fases_por_estrategia = {}
    todas_fases = set(serial_fases.keys())
    for estrategia in estrategias:
        fases = ler_fases_p_medianos(os.path.join(BASE_DIR, estrategia, CSV_NAME))
        fases_por_estrategia[estrategia] = fases
        todas_fases.update(fases.keys())
    # Garantir que P5_COPY_HOST esteja presente
    todas_fases.add('P5_COPY_HOST')
    # Montar header
    header = ['PHASE'] + [e.replace(suffix, '') for e in estrategias]
    # Montar linhas
    linhas = []
    for fase in sorted(todas_fases):
        if fase == 'PIXEL_FILTER':
            continue  # Garantia extra
        linha = [fase]
        for estrategia in estrategias:
            base = serial_fases.get(fase)
            valor = fases_por_estrategia[estrategia].get(fase)
            if estrategia == serial_dir:
                # Para serial, se não existir a fase, deixa vazio
                speedup = 1.0 if valor is not None else ''
            else:
                if valor is None or base is None or valor == 0:
                    speedup = ''
                else:
                    speedup = round(base / valor, 2)
            linha.append(speedup)
        linhas.append(linha)
    # Escrever CSV
    with open(os.path.join(BASE_DIR, output_csv), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(linhas)
    print(f'Arquivo {output_csv} gerado com sucesso em {BASE_DIR}.')

if __name__ == '__main__':
    gerar_speedup_csv(STEPP_SUFFIX, 'speedup_steep.csv')
    gerar_speedup_csv(SEBAL_SUFFIX, 'speedup_sebal.csv') 