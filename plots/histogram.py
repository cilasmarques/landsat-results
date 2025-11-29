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
    import re
    df = df.copy()

    rows = []
    for approach in df['approach'].drop_duplicates():
        al = approach.lower()

        # algoritmo
        if 'sebal' in al:
            algorithm = 'SEBAL'
        elif 'steep' in al:
            algorithm = 'STEEP'
        else:
            algorithm = 'UNKNOWN'

        # tipo
        if 'serial' in al:
            approach_type = 'Serial'
        elif 'kernels' in al:
            approach_type = 'Kernels'
        else:
            approach_type = 'Other'

        # precisão
        if 'double' in al:
            precision = 'Double'
        elif 'float' in al:
            precision = 'Float'
        else:
            precision = 'Unknown'

        # tipo de memória (evita casar o 'r' de "kernels")
        def has_token(token):
            return re.search(rf'(^|[-_]){token}($|[-_])', al) is not None

        if has_token('fm'):
            memory_type = 'FM'
        elif has_token('st'):
            memory_type = 'ST'
        elif has_token('r'):
            memory_type = 'R'
        else:
            memory_type = 'Unknown'

        rows.append({
            'approach': approach,
            'algorithm': algorithm,
            'approach_type': approach_type,
            'precision': precision,
            'memory_type': memory_type,
            'approach_label': f"{approach_type}\n({precision}, {memory_type})"
        })

    info_df = pd.DataFrame(rows)

    # many_to_one garante que info_df tem uma linha por 'approach'
    df = df.merge(info_df, on='approach', how='left', validate='many_to_one')

    return df

def create_individual_histograms(data, output_dir):
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
        
        # Obter informações da abordagem
        algorithm = approach_data['algorithm'].iloc[0]
        approach_type = approach_data['approach_type'].iloc[0]
        precision = approach_data['precision'].iloc[0]
        memory_type = approach_data['memory_type'].iloc[0]
        
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
        
        p = (ggplot(approach_data, aes(x='time_seconds', fill='algorithm'))
            + geom_histogram(alpha=0.6, bins=20, position='identity')
            + scale_fill_manual(values={'STEEP': '#00bdbd', 'SEBAL': '#fc3a3a'})
            + labs(title=f'Histograma — Tempo Total por Experimento — {approach}',
                    x='Tempo Total (s)',
                    y='Contagem')
            + theme_bw()
            + theme(
                axis_text_x=element_text(size=14),
                axis_text_y=element_text(size=12),
                axis_title_x=element_text(size=16),
                axis_title_y=element_text(size=16),
                plot_title=element_text(size=18, hjust=0.5),
                legend_title=element_text(size=14),
                legend_text=element_text(size=12),
                figure_size=(8, 6)
            ))

        filename = f"histogram_individual_{approach}.png"
        p.save(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"    Salvo: {output_dir / filename}")

def main():
    """Função principal"""
    input_dir = Path('results') 
    output_dir = Path('images/histograms')
    output_dir.mkdir(exist_ok=True)
    
    print("=== GERAÇÃO DE HISTOGRAMAS ===")
    print(f"Diretório de entrada: {input_dir}")
    print(f"Diretório de saída: {output_dir}")
    
    # Carregar dados de todas as abordagens
    try:
        all_data = load_all_approach_data(input_dir)
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return
    
    print("\n=== GERANDO HISTOGRAMAS INDIVIDUAIS ===")
    create_individual_histograms(all_data, output_dir)
        
    print("\n=== GERAÇÃO CONCLUÍDA ===")
    print(f"Gráficos de histogramas salvos em: {output_dir}")

if __name__ == "__main__":
    main()
