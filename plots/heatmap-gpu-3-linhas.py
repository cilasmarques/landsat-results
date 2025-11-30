#!/usr/bin/env python3
"""
Script para gerar heatmaps comparando estratégias GPU específicas
Baseado no create_macrogroup_heatmap.py mas focado nas estratégias GPU
"""

import pandas as pd
import numpy as np
from pathlib import Path
from plotnine import *
import warnings
warnings.filterwarnings('ignore')

# Estratégias GPU para comparação
GPU_STRATEGIES = [
    'kernels-double-fm-r-sebal', 'kernels-double-fm-r-steep',
    'kernels-double-fm-s-sebal', 'kernels-double-fm-s-steep',
    'kernels-float-fm-s-sebal', 'kernels-float-fm-s-steep'
]

# Mapeamento para nomes das abordagens
STRATEGY_LABELS = {
    'kernels-double-fm-r-sebal': 'GPU',
    'kernels-double-fm-r-steep': 'GPU',
    'kernels-double-fm-s-sebal': 'GPU leitura paralela',
    'kernels-double-fm-s-steep': 'GPU leitura paralela',
    'kernels-float-fm-s-sebal': 'GPU Float',
    'kernels-float-fm-s-steep': 'GPU Float'
}

def load_macrogroup_data(input_dir):
    """Carrega dados das etapas do diretório summarized_results_grouped"""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {input_path}")
    
    all_data = []
    
    # Listar apenas as estratégias GPU especificadas
    strategy_dirs = [d for d in input_path.iterdir() if d.is_dir() and not d.name.startswith('.') and d.name in GPU_STRATEGIES]
    
    print(f"Carregando dados de {len(strategy_dirs)} estratégias GPU...")
    
    for strategy_dir in strategy_dirs:
        strategy_name = strategy_dir.name
        final_time_file = strategy_dir / 'final-time-grouped.csv'
        
        if not final_time_file.exists():
            print(f"  Aviso: Arquivo não encontrado: {final_time_file}")
            continue
        
        # Carregar dados
        df = pd.read_csv(final_time_file)
        df['strategy_algorithm'] = strategy_name
        all_data.append(df)
        print(f"  Carregado: {strategy_name} ({len(df)} registros)")
    
    if not all_data:
        raise ValueError("Nenhum dado foi carregado")
    
    # Combinar todos os dados
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total de registros carregados: {len(combined_df)}")
    
    return combined_df

def extract_strategy_and_algorithm(df):
    """Extrai estratégia e algoritmo da coluna strategy_algorithm"""
    df = df.copy()
    
    # Mapear strategy_algorithm para strategy e algorithm
    strategy_mapping = {}
    algorithm_mapping = {}
    
    for _, row in df.iterrows():
        strategy_alg = row['strategy_algorithm']
        
        # Determinar algorithm (SEBAL ou STEEP)
        if 'sebal' in strategy_alg.lower():
            algorithm_mapping[strategy_alg] = 'SEBAL'
        elif 'steep' in strategy_alg.lower():
            algorithm_mapping[strategy_alg] = 'STEEP'
        else:
            algorithm_mapping[strategy_alg] = 'UNKNOWN'
        
        # Determinar strategy usando o mapeamento
        strategy_mapping[strategy_alg] = STRATEGY_LABELS.get(strategy_alg, 'OTHER')
    
    df['algorithm'] = df['strategy_algorithm'].map(algorithm_mapping)
    df['strategy'] = df['strategy_algorithm'].map(strategy_mapping)
    df['strategy_plot'] = df['strategy']
    
    return df

def _prepare_macrogroups_data(macrogroup_data):
    """Prepara dados das etapas para comparação GPU"""
    # Extrair strategy/algorithm
    df = extract_strategy_and_algorithm(macrogroup_data)
    
    # Mapeamento das etapas para português
    phase_mapping = {
        'E_READ': 'Leitura dos\ndados de entrada',
        'E0_MED_RAD': 'Medidas\nradiométricas',
        'E1_ID_VEG': 'Índices sobre\na vegetação',
        'E2_VAR_T': 'Variáveis\ntérmicas',
        'E3_MED_RN_G': 'Medidas de\nradiação',
        'E4_PIX': 'Seleção de\npixels',
        'E5_VAR_AR': 'Variáveis\naerodinâmicas',
        'E6_EVAPO': 'Evapotranspiração',
        'E_WRITE': 'Escrita dos\ndados de saída',
        'E_TOTAL': 'Tempo\nTotal'
    }
    
    phase_order_pt = [
        'Leitura dos\ndados de entrada', 'Medidas\nradiométricas', 'Índices sobre\na vegetação',
        'Variáveis\ntérmicas', 'Medidas de\nradiação', 'Seleção de\npixels',
        'Variáveis\naerodinâmicas', 'Evapotranspiração', 'Escrita dos\ndados de saída', 'Tempo\nTotal'
    ]
    
    return df, phase_mapping, phase_order_pt

def _build_heatmap_df(df, algorithm, phase_mapping, phase_order_pt, y_key='strategy_plot'):
    """Monta o dataframe do heatmap com LABEL tempo (perc) e ordenação de fases."""
    df_alg = df[df['algorithm'] == algorithm].copy()
    if df_alg.empty:
        return None

    # Totais por linha (para percentual)
    totals = df_alg[df_alg['PHASE'] == 'E_TOTAL'].groupby([y_key])['TIMESTAMP_median'].mean().reset_index()
    totals = totals.rename(columns={'TIMESTAMP_median': 'TOTAL_SEC'})

    pivot = df_alg.pivot_table(values='TIMESTAMP_median', index=y_key, columns='PHASE', aggfunc='mean')
    heatmap = pivot.reset_index().melt(id_vars=[y_key], var_name='PHASE', value_name='TIMESTAMP_median')
    heatmap = heatmap.merge(totals, on=y_key, how='left')

    heatmap['PHASE_PT'] = heatmap['PHASE'].map(phase_mapping)
    heatmap['PHASE_PT'] = pd.Categorical(heatmap['PHASE_PT'], categories=phase_order_pt, ordered=True)
    heatmap = heatmap.dropna(subset=['TIMESTAMP_median', 'PHASE_PT', 'TOTAL_SEC'])

    heatmap['PERC'] = (heatmap['TIMESTAMP_median'] / heatmap['TOTAL_SEC']) * 100.0
    heatmap.loc[heatmap['PHASE'] == 'E_TOTAL', 'PERC'] = 100.0

    # Formatação: converter de milissegundos para segundos, 3 casas decimais, percentual 2 casas decimais, vírgula
    heatmap['TEMPO_STR'] = heatmap['TIMESTAMP_median'].apply(lambda x: f"{x/1000:.3f}".replace('.', ','))
    heatmap['PERC_STR'] = heatmap['PERC'].apply(lambda x: f"{x:.2f}".replace('.', ','))
    heatmap['LABEL'] = heatmap['TEMPO_STR'] + '\n(' + heatmap['PERC_STR'] + '%)'
    return heatmap

def create_gpu_comparison_heatmap(macrogroup_data, output_dir):
    """Heatmap para comparação de todas as estratégias GPU"""
    df, phase_mapping, phase_order_pt = _prepare_macrogroups_data(macrogroup_data)
    
    if df.empty:
        print("  Aviso: Sem dados encontrados para comparação GPU")
        return
    
    # Criar heatmaps para SEBAL
    print("  1. Comparação GPU SEBAL...")
    df_sebal = df[df['algorithm'] == 'SEBAL'].copy()
    
    if df_sebal.empty:
        print("    Aviso: Sem dados SEBAL encontrados")
    else:
        sebal_heatmap = _build_heatmap_df(df_sebal, 'SEBAL', phase_mapping, phase_order_pt, y_key='strategy_plot')
        
        if sebal_heatmap is not None and not sebal_heatmap.empty:
            # Definir ordem dos plots (invertida para que GPU apareça no topo)
            plot_order = ['GPU Float', 'GPU leitura paralela', 'GPU']
            sebal_heatmap['strategy_plot'] = pd.Categorical(sebal_heatmap['strategy_plot'], 
                                                           categories=plot_order, ordered=True)
            
            # Criar o plot
            p = (ggplot(sebal_heatmap, aes(x='PHASE_PT', y='strategy_plot', fill='TIMESTAMP_median')) +
                 geom_tile() +
                 geom_text(aes(label='LABEL'), size=12, color='black', ha='center') +
                 scale_fill_gradient(low='#F5F5F5', high='#FF6B6B', name='Tempo (s)') +
                 labs(title='Heatmap de Tempos - SEBAL (Comparação GPU)', 
                      x='Etapa', y='Estratégia') +
                 theme_bw() +
                 theme(axis_text_x=element_text(rotation=45, hjust=1, size=14),
                       axis_text_y=element_text(size=14),
                       axis_title_x=element_text(size=16),
                       axis_title_y=element_text(size=16),
                       plot_title=element_text(size=18),
                       legend_text=element_text(size=14),
                       legend_title=element_text(size=16),
                       figure_size=(12, 4)))
            
            filename = "gpu_comparison_sebal.png"
            p.save(output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"    Salvo: {output_dir / filename}")
    
    # Criar heatmaps para STEEP
    print("  2. Comparação GPU STEEP...")
    df_steep = df[df['algorithm'] == 'STEEP'].copy()
    
    if df_steep.empty:
        print("    Aviso: Sem dados STEEP encontrados")
    else:
        steep_heatmap = _build_heatmap_df(df_steep, 'STEEP', phase_mapping, phase_order_pt, y_key='strategy_plot')
        
        if steep_heatmap is not None and not steep_heatmap.empty:
            # Definir ordem dos plots (invertida para que GPU apareça no topo)
            plot_order = ['GPU Float', 'GPU leitura paralela', 'GPU']
            steep_heatmap['strategy_plot'] = pd.Categorical(steep_heatmap['strategy_plot'], 
                                                           categories=plot_order, ordered=True)
            
            # Criar o plot
            p = (ggplot(steep_heatmap, aes(x='PHASE_PT', y='strategy_plot', fill='TIMESTAMP_median')) +
                 geom_tile() +
                 geom_text(aes(label='LABEL'), size=12, color='black', ha='center') +
                 scale_fill_gradient(low='#F5F5F5', high='#4ECDC4', name='Tempo (s)') +
                 labs(title='Heatmap de Tempos - STEEP (Comparação GPU)', 
                      x='Etapa', y='Estratégia') +
                 theme_bw() +
                 theme(axis_text_x=element_text(rotation=45, hjust=1, size=14),
                       axis_text_y=element_text(size=14),
                       axis_title_x=element_text(size=16),
                       axis_title_y=element_text(size=16),
                       plot_title=element_text(size=18),
                       legend_text=element_text(size=14),
                       legend_title=element_text(size=16),
                       figure_size=(12, 4)))
            
            filename = "gpu_comparison_steep.png"
            p.save(output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"    Salvo: {output_dir / filename}")
    
    # Criar heatmap combinado com todos os algoritmos
    print("  3. Comparação GPU combinada (SEBAL + STEEP)...")
    
    # Combinar dados de ambos os algoritmos
    all_heatmaps = []
    
    for algorithm in ['SEBAL', 'STEEP']:
        df_alg = df[df['algorithm'] == algorithm].copy()
        if not df_alg.empty:
            heatmap = _build_heatmap_df(df_alg, algorithm, phase_mapping, phase_order_pt, y_key='strategy_plot')
            if heatmap is not None and not heatmap.empty:
                heatmap['algorithm'] = algorithm
                heatmap['combined_label'] = heatmap['strategy_plot'] + f' ({algorithm})'
                all_heatmaps.append(heatmap)
    
    if all_heatmaps:
        combined_heatmap = pd.concat(all_heatmaps, ignore_index=True)
        
        # Definir ordem dos plots (invertida para que GPU apareça no topo)
        plot_order = [
            'GPU Float (SEBAL)', 'GPU Float (STEEP)',
            'GPU leitura paralela (SEBAL)', 'GPU leitura paralela (STEEP)',
            'GPU (SEBAL)', 'GPU (STEEP)'
        ]
        combined_heatmap['combined_label'] = pd.Categorical(combined_heatmap['combined_label'], 
                                                           categories=plot_order, ordered=True)
        
        # Criar o plot combinado
        p = (ggplot(combined_heatmap, aes(x='PHASE_PT', y='combined_label', fill='TIMESTAMP_median')) +
             geom_tile() +
             geom_text(aes(label='LABEL'), size=10, color='black', ha='center') +
             scale_fill_gradient(low='#F5F5F5', high='#FF8C00', name='Tempo (s)') +
             labs(title='Heatmap de Tempos - Comparação GPU (SEBAL + STEEP)', 
                  x='Etapa', y='Estratégia - Algoritmo') +
             theme_bw() +
             theme(axis_text_x=element_text(rotation=45, hjust=1, size=14),
                   axis_text_y=element_text(size=12),
                   axis_title_x=element_text(size=16),
                   axis_title_y=element_text(size=16),
                   plot_title=element_text(size=18),
                   legend_text=element_text(size=14),
                   legend_title=element_text(size=16),
                   figure_size=(12, 6)))
        
        filename = "gpu_comparison_combined.png"
        p.save(output_dir / filename, dpi=300, bbox_inches='tight')
        print(f"    Salvo: {output_dir / filename}")

def main():
    """Função principal"""
    input_dir = Path('summarized-groups')
    output_dir = Path('images/heatmaps-3-linhas')
    
    # Criar diretório de saída se não existir
    output_dir.mkdir(exist_ok=True)
    
    print("=== GERAÇÃO DE HEATMAPS DE COMPARAÇÃO GPU ===")
    print(f"Diretório de entrada: {input_dir}")
    print(f"Diretório de saída: {output_dir}")
    
    # Carregar dados das etapas
    try:
        macrogroup_data = load_macrogroup_data(input_dir)
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return
    
    print("\n=== GERANDO HEATMAPS DE COMPARAÇÃO GPU ===")
    
    # Heatmap de comparação GPU
    create_gpu_comparison_heatmap(macrogroup_data, output_dir)
    
    print("\n=== GERAÇÃO CONCLUÍDA ===")
    print(f"Heatmaps salvos em: {output_dir}")

if __name__ == "__main__":
    main()
