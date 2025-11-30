#!/usr/bin/env python3
"""
Script para gerar gráficos de barras com intervalos de confiança 95%
usando dados brutos P_TOTAL de cada abordagem
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

def create_individual_barplots_with_ci(data, output_dir):
    """Cria gráficos de barras individuais com intervalos de confiança 95%"""
    df = extract_approach_info(data)
    
    # Converter tempo para segundos (dados brutos já são P_TOTAL)
    df['time_seconds'] = df['TIMESTAMP'] / 1000
    
    # Remover outliers
    df_clean = remove_outliers_iqr(df, 'time_seconds', 'approach')
    
    print(f"Dados originais: {len(df)} registros")
    print(f"Dados após remoção de outliers: {len(df_clean)} registros")
    
    # Criar gráfico de barras individual para cada abordagem
    for approach in df_clean['approach'].unique():
        approach_data = df_clean[df_clean['approach'] == approach]
        
        if approach_data.empty:
            continue
        
        # Calcular estatísticas para barras e intervalos de confiança
        stats_data = []
        for alg in approach_data['algorithm'].unique():
            alg_data = approach_data[approach_data['algorithm'] == alg]
            mean_time = alg_data['time_seconds'].mean()
            std_time = alg_data['time_seconds'].std()
            count = len(alg_data)
            
            # Intervalo de confiança 95% (aproximação normal)
            ci_95 = 1.96 * (std_time / np.sqrt(count))
            
            stats_data.append({
                'algorithm': alg,
                'mean_time': mean_time,
                'std_time': std_time,
                'ci_95': ci_95,
                'count': count,
                'ymin': mean_time - ci_95,
                'ymax': mean_time + ci_95
            })
        
        stats_df = pd.DataFrame(stats_data)
        
        # Criar gráfico de barras com intervalos de confiança
        p = (ggplot(stats_df, aes(x='algorithm', y='mean_time', fill='algorithm')) +
             geom_bar(stat='identity', alpha=0.7, width=0.6) +
             # Adicionar barras de erro para intervalos de confiança
             geom_errorbar(stats_df, aes(x='algorithm', y='mean_time', ymin='ymin', ymax='ymax'), 
                          width=0.2, size=1, color='red') +
             scale_fill_manual(values=['#1f77b4', '#ff7f0e'], name='Algoritmo') +
             labs(title=f'Gráfico de Barras - Tempo Total - {approach}',
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
        
        filename = f"barplot_individual_{approach}.png"
        p.save(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"    Salvo: {output_dir / filename}")

def create_combined_barplot_with_ci(data, output_dir):
    """Cria gráfico de barras combinado com intervalos de confiança 95%"""
    df = extract_approach_info(data)
    
    # Converter tempo para segundos (dados brutos já são P_TOTAL)
    df['time_seconds'] = df['TIMESTAMP'] / 1000
    
    # Remover outliers
    df_clean = remove_outliers_iqr(df, 'time_seconds', 'approach')
    
    # Calcular estatísticas para barras e intervalos de confiança
    combined_stats_data = []
    for approach in df_clean['approach'].unique():
        approach_data = df_clean[df_clean['approach'] == approach]
        mean_time = approach_data['time_seconds'].mean()
        std_time = approach_data['time_seconds'].std()
        count = len(approach_data)
        algorithm = approach_data['algorithm'].iloc[0]
        
        # Intervalo de confiança 95%
        ci_95 = 1.96 * (std_time / np.sqrt(count))
        
        combined_stats_data.append({
            'approach': approach,
            'algorithm': algorithm,
            'mean_time': mean_time,
            'std_time': std_time,
            'ci_95': ci_95,
            'count': count,
            'ymin': mean_time - ci_95,
            'ymax': mean_time + ci_95
        })
    
    combined_stats_df = pd.DataFrame(combined_stats_data)
    
    # Definir ordem personalizada das abordagens
    approach_order = [
        'serial-double-r-sebal',
        'serial-double-r-steep',
        'kernels-double-fm-r-sebal',
        'kernels-double-fm-r-steep',
        'kernels-double-fm-s-sebal',
        'kernels-double-fm-s-steep',
        'kernels-float-fm-s-sebal',
        'kernels-float-fm-s-steep'
    ]
    
    # Converter approach para Categorical com ordem personalizada
    combined_stats_df['approach'] = pd.Categorical(combined_stats_df['approach'], 
                                                   categories=approach_order, ordered=True)
    
    # Criar gráfico de barras combinado com intervalos de confiança
    p = (ggplot(combined_stats_df, aes(x='approach', y='mean_time', fill='algorithm')) +
         geom_bar(stat='identity', alpha=0.7, width=0.6) +
         # Adicionar barras de erro para intervalos de confiança
         geom_errorbar(combined_stats_df, aes(x='approach', y='mean_time', ymin='ymin', ymax='ymax'), 
                      width=0.1, size=0.8, color='red') +
         scale_fill_manual(values=['#1f77b4', '#ff7f0e'], name='Algoritmo') +
         labs(title='Gráfico de Barras - Tempos Totais por Abordagem',
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
    
    filename = "barplot_combined_total.png"
    p.save(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"    Salvo: {output_dir / filename}")

def main():
    """Função principal"""
    input_dir = Path('results') 
    output_dir = Path('images/barplots-with-confidence-intervals')
    output_dir.mkdir(exist_ok=True)
    
    print("=== GERAÇÃO DE GRÁFICOS DE BARRAS COM INTERVALOS DE CONFIANÇA ===")
    print(f"Diretório de entrada: {input_dir}")
    print(f"Diretório de saída: {output_dir}")
    
    # Carregar dados de todas as abordagens
    try:
        all_data = load_all_approach_data(input_dir)
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return
    
    print("\n=== GERANDO GRÁFICOS DE BARRAS INDIVIDUAIS ===")
    create_individual_barplots_with_ci(all_data, output_dir)
    
    print("\n=== GERANDO GRÁFICO DE BARRAS COMBINADO ===")
    create_combined_barplot_with_ci(all_data, output_dir)
    
    print("\n=== GERAÇÃO CONCLUÍDA ===")
    print(f"Gráficos de barras salvos em: {output_dir}")

if __name__ == "__main__":
    main()
