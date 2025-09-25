#!/usr/bin/env python3
"""
Script para fazer summarize results 2
Agrupa os tempos totais do summarized_results em categorias específicas
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

def load_and_group_data(input_dir, output_dir):
    """Carrega dados do summarized_results e agrupa nas categorias especificadas"""
    
    # Definir as categorias e suas fases correspondentes
    categories = {
        'E_READ': ['P0_READ_INPUT'],
        'E0_MED_RAD': ['RADIANCE', 'REFLECTANCE', 'ALBEDO'],
        'E1_ID_VEG': ['NDVI', 'LAI', 'PAI'],
        'E2_VAR_T': ['ENB_EMISSIVITY', 'EO_EMISSIVITY', 'EA_EMISSIVITY', 'SURFACE_TEMPERATURE'],
        'E3_MED_RN_G': ['SHORT_WAVE_RADIATION', 'LARGE_WAVES_RADIATION', 'NET_RADIATION', 'SOIL_HEAT_FLUX'],
        'E4_PIX': ['P2_PIXEL_SEL'],
        'E5_VAR_AR': ['D0', 'ZOM', 'USTAR', 'KB1', 'RAH_INI', 'RAH_CYCLE', 'SENSIBLE_HEAT_FLUX'],
        'E6_EVAPO': ['LATENT_HEAT_FLUX', 'NET_RADIATION_24H', 'EVAPOTRANSPIRATION_24H'],
        'E_WRITE': ['P5_COPY_HOST', 'P6_SAVE_PRODS'],
        'E_TOTAL': ['P_TOTAL']
    }
    
    # Criar diretório de saída se não existir
    output_dir.mkdir(exist_ok=True)
    
    # Listar todas as pastas de estratégias
    strategy_dirs = [d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"Processando {len(strategy_dirs)} estratégias...")
    
    for strategy_dir in strategy_dirs:
        strategy_name = strategy_dir.name
        print(f"Processando: {strategy_name}")
        
        # Caminho para o arquivo final-time.csv
        final_time_file = strategy_dir / 'final-time.csv'
        
        if not final_time_file.exists():
            print(f"  Arquivo não encontrado: {final_time_file}")
            continue
        
        # Carregar dados
        df = pd.read_csv(final_time_file)
        
        # Lista para armazenar os dados agrupados
        grouped_data = []
        
        # Processar cada categoria
        for category, phases in categories.items():
            # Filtrar fases que existem nos dados
            available_phases = [phase for phase in phases if phase in df['PHASE'].values]
            
            if not available_phases:
                print(f"  Aviso: Nenhuma fase encontrada para {category} em {strategy_name}")
                continue
            
            # Filtrar dados para as fases da categoria
            category_data = df[df['PHASE'].isin(available_phases)]
            
            if category_data.empty:
                continue
            
            # Calcular estatísticas agrupadas
            total_median = category_data['TIMESTAMP_median'].sum()
            total_mean = category_data['TIMESTAMP_mean'].sum()
            total_std = np.sqrt((category_data['TIMESTAMP_std']**2).sum())  # Soma das variâncias
            total_min = category_data['TIMESTAMP_min'].sum()
            total_max = category_data['TIMESTAMP_max'].sum()
            total_count = category_data['TIMESTAMP_count'].iloc[0]  # Mesmo count para todas
            
            # Obter a estratégia (primeira linha)
            strategy = category_data['STRATEGY'].iloc[0]
            
            # Criar linha de dados agrupados
            grouped_row = {
                'STRATEGY': strategy,
                'PHASE': category,
                'TIMESTAMP_count': total_count,
                'TIMESTAMP_median': total_median,
                'TIMESTAMP_mean': total_mean,
                'TIMESTAMP_std': total_std,
                'TIMESTAMP_min': total_min,
                'TIMESTAMP_max': total_max,
                'strategy_algorithm': strategy_name,
                'phases_included': ', '.join(available_phases)
            }
            
            grouped_data.append(grouped_row)
        
        # Criar DataFrame com dados agrupados
        if grouped_data:
            grouped_df = pd.DataFrame(grouped_data)
            
            # Criar diretório para a estratégia
            strategy_output_dir = output_dir / strategy_name
            strategy_output_dir.mkdir(exist_ok=True)
            
            # Salvar arquivo agrupado
            output_file = strategy_output_dir / 'final-time-grouped.csv'
            grouped_df.to_csv(output_file, index=False)
            
            print(f"  Salvo: {output_file}")
            print(f"  Categorias processadas: {len(grouped_data)}")
            
            # Mostrar resumo das categorias
            for _, row in grouped_df.iterrows():
                print(f"    {row['PHASE']}: {row['TIMESTAMP_median']:.3f}s ({row['phases_included']})")
        else:
            print(f"  Nenhum dado agrupado gerado para {strategy_name}")

def main():
    """Função principal"""
    input_dir = Path('summarized-results')
    output_dir = Path('summarized-groups')
    
    if not input_dir.exists():
        print(f"Erro: Diretório {input_dir} não encontrado!")
        return
    
    print("=== SUMMARIZE RESULTS 2 ===")
    print(f"Diretório de entrada: {input_dir}")
    print(f"Diretório de saída: {output_dir}")
    print()
    
    load_and_group_data(input_dir, output_dir)
    
    print("\n=== PROCESSAMENTO CONCLUÍDO ===")
    print(f"Resultados salvos em: {output_dir}")

if __name__ == '__main__':
    main()
