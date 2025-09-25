#!/usr/bin/env python3
"""
Script para gerar boxplots por abordagem usando dados de final-time-grouped.csv
Cada boxplot mostra a distribuição dos tempos para cada fase de cada abordagem
"""

import pandas as pd
import numpy as np
from pathlib import Path
from plotnine import *
import warnings
warnings.filterwarnings('ignore')

def remove_outliers_iqr(df, column='time_seconds', groupby='approach'):
    """Remove outliers usando o método IQR (Interquartile Range)"""
    df_clean = df.copy()
    
    # Para cada abordagem, remover outliers
    for approach in df[groupby].unique():
        mask = df[groupby] == approach
        data = df.loc[mask, column]
        
        # Calcular Q1, Q3 e IQR
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        # Definir limites para outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remover outliers
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        df_clean.loc[mask & outlier_mask, column] = np.nan
    
    # Remover linhas com NaN
    df_clean = df_clean.dropna(subset=[column])
    
    return df_clean

def load_all_approach_data(input_dir):
    """Carrega dados brutos de todas as abordagens do diretório results_raw"""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {input_path}")
    
    all_data = []
    
    # Listar todos os diretórios de abordagens
    approach_dirs = [d for d in input_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"Carregando dados brutos de {len(approach_dirs)} abordagens...")
    
    for approach_dir in approach_dirs:
        approach_name = approach_dir.name
        print(f"  Processando: {approach_name}")
        
        # Listar todos os experimentos
        experiment_dirs = [d for d in approach_dir.iterdir() if d.is_dir() and d.name.startswith('experiment')]
        
        approach_data = []
        for experiment_dir in experiment_dirs:
            time_file = experiment_dir / 'time.csv'
            
            if not time_file.exists():
                print(f"    Aviso: Arquivo não encontrado: {time_file}")
                continue
            
            try:
                # Carregar dados do experimento
                df = pd.read_csv(time_file)
                df['approach'] = approach_name
                df['experiment'] = experiment_dir.name
                
                # Filtrar apenas P_TOTAL
                p_total_data = df[df['PHASE'] == 'P_TOTAL'].copy()
                
                if not p_total_data.empty:
                    approach_data.append(p_total_data)
                    
            except Exception as e:
                print(f"    Erro ao carregar {time_file}: {e}")
                continue
        
        if approach_data:
            # Combinar dados de todos os experimentos desta abordagem
            combined_approach_data = pd.concat(approach_data, ignore_index=True)
            all_data.append(combined_approach_data)
            print(f"    Carregado: {len(approach_data)} experimentos ({len(combined_approach_data)} registros P_TOTAL)")
        else:
            print(f"    Nenhum dado P_TOTAL encontrado para {approach_name}")
    
    if not all_data:
        raise ValueError("Nenhum dado foi carregado")
    
    # Combinar todos os dados
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total de registros P_TOTAL carregados: {len(combined_df)}")
    
    return combined_df

def extract_approach_info(df):
    """Extrai informações da abordagem (tipo, algoritmo, precisão)"""
    df = df.copy()
    
    # Mapear approach para informações estruturadas
    approach_info = []
    
    for _, row in df.iterrows():
        approach = row['approach']
        
        # Determinar algoritmo
        if 'sebal' in approach.lower():
            algorithm = 'SEBAL'
        elif 'steep' in approach.lower():
            algorithm = 'STEEP'
        else:
            algorithm = 'UNKNOWN'
        
        # Determinar tipo de abordagem
        if 'serial' in approach.lower():
            approach_type = 'Serial'
        elif 'kernels' in approach.lower():
            approach_type = 'Kernels'
        else:
            approach_type = 'Other'
        
        # Determinar precisão
        if 'double' in approach.lower():
            precision = 'Double'
        elif 'float' in approach.lower():
            precision = 'Float'
        else:
            precision = 'Unknown'
        
        # Determinar tipo de memória
        if 'fm' in approach.lower():
            memory_type = 'FM'
        elif 'r' in approach.lower():
            memory_type = 'R'
        elif 'st' in approach.lower():
            memory_type = 'ST'
        else:
            memory_type = 'Unknown'
        
        approach_info.append({
            'approach': approach,
            'algorithm': algorithm,
            'approach_type': approach_type,
            'precision': precision,
            'memory_type': memory_type,
            'approach_label': f"{approach_type}\n({precision}, {memory_type})"
        })
    
    # Adicionar informações ao DataFrame
    info_df = pd.DataFrame(approach_info)
    df = df.merge(info_df, on='approach', how='left')
    
    return df

def create_individual_boxplots_by_approach(data, output_dir):
    """Cria boxplot individual para cada abordagem usando dados brutos P_TOTAL sem outliers"""
    df = extract_approach_info(data)
    
    # Converter tempo para segundos (dados brutos já são P_TOTAL)
    df['time_seconds'] = df['TIMESTAMP'] / 1000
    
    # Remover outliers
    df_clean = remove_outliers_iqr(df, 'time_seconds', 'approach')
    
    print(f"Dados originais: {len(df)} registros")
    print(f"Dados após remoção de outliers: {len(df_clean)} registros")
    
    # Criar boxplot individual para cada abordagem
    for approach in df_clean['approach'].unique():
        approach_data = df_clean[df_clean['approach'] == approach]
        
        if approach_data.empty:
            continue
        
        # Obter informações da abordagem
        algorithm = approach_data['algorithm'].iloc[0]
        approach_type = approach_data['approach_type'].iloc[0]
        precision = approach_data['precision'].iloc[0]
        memory_type = approach_data['memory_type'].iloc[0]
        
        
        # Criar boxplot limpo sem outliers
        p = (ggplot(approach_data, aes(x='algorithm', y='time_seconds', fill='algorithm')) +
             geom_boxplot(alpha=0.7, outlier_alpha=0) +  # Sem outliers
             scale_fill_manual(values=['#1f77b4', '#ff7f0e'], name='Algoritmo') +
             labs(title=f'Boxplot de Tempo Total - {approach}',
                  x='Algoritmo',
                  y='Tempo Total (s)',
                  fill='Algoritmo') +
             theme_bw() +
             theme(axis_text_x=element_text(size=14),
                   axis_text_y=element_text(size=12),
                   axis_title_x=element_text(size=16),
                   axis_title_y=element_text(size=16),
                   plot_title=element_text(size=18, hjust=0.5),
                   legend_text=element_text(size=12),
                   legend_title=element_text(size=14),
                   figure_size=(8, 6)))
        
        filename = f"boxplot_individual_{approach}.png"
        p.save(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"    Salvo: {output_dir / filename}")

def create_combined_boxplot(data, output_dir):
    """Cria boxplot combinado mostrando todas as abordagens usando dados brutos sem outliers"""
    df = extract_approach_info(data)
    
    # Converter tempo para segundos (dados brutos já são P_TOTAL)
    df['time_seconds'] = df['TIMESTAMP'] / 1000
    
    # Remover outliers
    df_clean = remove_outliers_iqr(df, 'time_seconds', 'approach')
    
    # Definir ordem personalizada das abordagens
    approach_order = [
        'serial-double-r-sebal',
        'serial-double-r-steep',
        'kernels-double-fm-r-sebal',
        'kernels-double-fm-r-steep',
        'kernels-double-fm-st-sebal',
        'kernels-double-fm-st-steep',
        'kernels-float-st-sebal',
        'kernels-float-st-steep'
    ]
    
    # Converter approach para Categorical com ordem personalizada
    df_clean['approach'] = pd.Categorical(df_clean['approach'], 
                                         categories=approach_order, ordered=True)
    
    # Criar boxplot combinado limpo sem outliers
    p = (ggplot(df_clean, aes(x='approach', y='time_seconds', fill='algorithm')) +
         geom_boxplot(alpha=0.7, outlier_alpha=0) +  # Sem outliers
         scale_fill_manual(values=['#1f77b4', '#ff7f0e'], name='Algoritmo') +
         labs(title='Boxplot de Tempos Totais por Abordagem',
              x='Abordagem',
              y='Tempo Total (s)',
              fill='Algoritmo') +
         theme_bw() +
         theme(axis_text_x=element_text(size=10, angle=45, hjust=1),
               axis_text_y=element_text(size=12),
               axis_title_x=element_text(size=14),
               axis_title_y=element_text(size=14),
               plot_title=element_text(size=16, hjust=0.5),
               legend_text=element_text(size=12),
               legend_title=element_text(size=14),
               figure_size=(15, 8)))
    
    filename = "boxplot_combined_total.png"
    p.save(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"    Salvo: {output_dir / filename}")


def main():
    """Função principal"""
    input_dir = Path('results')
    output_dir = Path('images/boxplots-by-approach')
    output_dir.mkdir(exist_ok=True)
    
    print("=== GERAÇÃO DE BOXPLOTS POR ABORDAGEM ===")
    print(f"Diretório de entrada: {input_dir}")
    print(f"Diretório de saída: {output_dir}")
    
    # Carregar dados de todas as abordagens
    try:
        all_data = load_all_approach_data(input_dir)
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return
    
    print("\n=== GERANDO BOXPLOTS INDIVIDUAIS POR ABORDAGEM ===")
    create_individual_boxplots_by_approach(all_data, output_dir)
    
    print("\n=== GERANDO BOXPLOT COMBINADO ===")
    create_combined_boxplot(all_data, output_dir)
    
    print("\n=== GERAÇÃO CONCLUÍDA ===")
    print(f"Boxplots salvos em: {output_dir}")

if __name__ == "__main__":
    main()
