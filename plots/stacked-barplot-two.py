#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
from plotnine import *
import warnings
warnings.filterwarnings('ignore')

# Estratégias para comparação CPU vs GPU
STRATEGIES = [
    'kernels-double-fm-st-steep',      # CPU
    'kernels-double-fm-r-steep'   # GPU
]

# Mapeamento para nomes das abordagens
STRATEGY_LABELS = {
    'kernels-double-fm-r-steep': 'SIMD',
    'kernels-double-fm-st-steep': 'Leitura\nOtimizada',
}

def load_macrogroup_data(input_dir):
    """Carrega dados das etapas do diretório summarized_results_grouped"""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {input_path}")
    
    all_data = []
    
    # Listar apenas as estratégias especificadas
    strategy_dirs = [d for d in input_path.iterdir() if d.is_dir() and not d.name.startswith('.') and d.name in STRATEGIES]
    
    print(f"Carregando dados de {len(strategy_dirs)} estratégias...")
    
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
        
        # Determinar algorithm (STEEP)
        if 'steep' in strategy_alg.lower():
            algorithm_mapping[strategy_alg] = 'Leitura\nOtimizada'
        else:
            algorithm_mapping[strategy_alg] = 'UNKNOWN'
        
        # Determinar strategy usando o mapeamento
        strategy_mapping[strategy_alg] = STRATEGY_LABELS.get(strategy_alg, 'OTHER')
    
    df['algorithm'] = df['strategy_algorithm'].map(algorithm_mapping)
    df['strategy'] = df['strategy_algorithm'].map(strategy_mapping)
    df['strategy_plot'] = df['strategy']
    
    return df

def prepare_three_phase_data(macrogroup_data):
    """Prepara dados para as três fases: Leitura, Cálculo e Escrita"""
    # Extrair strategy/algorithm
    df = extract_strategy_and_algorithm(macrogroup_data)
    
    # Fases de processamento para combinar no Processamento
    processing_phases = ['E0_MED_RAD', 'E1_ID_VEG', 'E2_VAR_T', 'E3_MED_RN_G', 'E4_PIX', 'E5_VAR_AR', 'E6_EVAPO']
    
    # Para cada estratégia e algoritmo, combinar as fases de processamento
    new_rows = []
    
    for (strategy, algorithm), group in df.groupby(['strategy_plot', 'algorithm']):
        # Obter dados das fases de processamento
        processing_data = group[group['PHASE'].isin(processing_phases)]
        
        if not processing_data.empty:
            # Somar os tempos das fases de processamento
            total_processing_time = processing_data['TIMESTAMP_median'].sum()
            
            # Criar nova linha para processamento combinado
            processing_row = processing_data.iloc[0].copy()
            processing_row['PHASE'] = 'E_CALCULO'
            processing_row['TIMESTAMP_median'] = total_processing_time
            new_rows.append(processing_row)
    
    # Adicionar as novas linhas de processamento
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    
    # Filtrar apenas as fases que queremos mostrar
    phases_to_keep = ['E_READ', 'E_READ_GPU', 'E_CALCULO', 'E_WRITE', 'E_WRITE_GPU']
    df = df[df['PHASE'].isin(phases_to_keep)]
    
    # Mapeamento das etapas para português
    phase_mapping = {
        'E_READ': 'Leitura dos dados\nde entrada',
        'E_READ_GPU': 'Leitura dos dados\nde entrada',
        'E_CALCULO': 'Processamento',
        'E_WRITE': 'Escrita dos dados\nde saída',
        'E_WRITE_GPU': 'Escrita dos dados\nde saída'
    }
    
    # Ordem das fases para o plot (leitura embaixo, escrita em cima)
    phase_order = [
        'Escrita dos dados\nde saída',
        'Processamento',
        'Leitura dos dados\nde entrada'
    ]
    
    return df, phase_mapping, phase_order

def create_cpu_vs_gpu_stacked_barplot_clean(macrogroup_data, output_dir):
    """Cria stacked barplot comparando CPU vs GPU SEM informações dentro das barras"""
    df, phase_mapping, phase_order = prepare_three_phase_data(macrogroup_data)
    
    if df.empty:
        print("  Aviso: Sem dados encontrados")
        return
    
    # Preparar dados para o plot - SOMANDO E_READ + E_READ_GPU e E_WRITE + E_WRITE_GPU
    plot_data = []
    
    for (strategy, algorithm), group in df.groupby(['strategy_plot', 'algorithm']):
        # Combinar E_READ e E_READ_GPU
        read_data = group[group['PHASE'].isin(['E_READ', 'E_READ_GPU'])]
        total_read_time = read_data['TIMESTAMP_median'].sum() if not read_data.empty else 0
        
        # Combinar E_WRITE e E_WRITE_GPU
        write_data = group[group['PHASE'].isin(['E_WRITE', 'E_WRITE_GPU'])]
        total_write_time = write_data['TIMESTAMP_median'].sum() if not write_data.empty else 0
        
        # Obter dados de cálculo
        calc_data = group[group['PHASE'] == 'E_CALCULO']
        total_calc_time = calc_data['TIMESTAMP_median'].sum() if not calc_data.empty else 0
        
        # Adicionar dados combinados
        if total_read_time > 0:
            plot_data.append({
                'Estratégia': strategy,
                'Algoritmo': algorithm,
                'Fase': 'Leitura dos dados\nde entrada',
                'Tempo (s)': total_read_time / 1000,
                'Tempo (ms)': total_read_time
            })
        
        if total_calc_time > 0:
            plot_data.append({
                'Estratégia': strategy,
                'Algoritmo': algorithm,
                'Fase': 'Processamento',
                'Tempo (s)': total_calc_time / 1000,
                'Tempo (ms)': total_calc_time
            })
        
        if total_write_time > 0:
            plot_data.append({
                'Estratégia': strategy,
                'Algoritmo': algorithm,
                'Fase': 'Escrita dos dados\nde saída',
                'Tempo (s)': total_write_time / 1000,
                'Tempo (ms)': total_write_time
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Converter para Categorical para manter a ordem
    plot_df['Fase'] = pd.Categorical(plot_df['Fase'], categories=phase_order, ordered=True)
    
    # A ordem da legenda deve ser INVERSA à ordem das barras
    # Isto garante que a primeira cor na legenda corresponde à barra no topo
    ordem_legenda = list(reversed(phase_order))
    
    # Paletas de cores por estratégia
    colors_cpu = {
        'Escrita dos dados\nde saída': '#bdbfbf',      # Cinza
        'Processamento': '#4ECDC4',                     # Azul médio
        'Leitura dos dados\nde entrada': '#00bdbd'     # Azul forte
    }
    
    colors_gpu = {
        'Escrita dos dados\nde saída': '#bdbfbf',      # Cinza
        'Processamento': '#4ECDC4',                     # Azul médio
        'Leitura dos dados\nde entrada': '#00bdbd'     # Azul forte
    }
    
    # Cores padrão para gráficos combinados
    colors = {
        'Escrita dos dados\nde saída': '#bdbfbf',      # Cinza
        'Processamento': '#4ECDC4',                     # Azul médio
        'Leitura dos dados\nde entrada': '#00bdbd'     # Azul forte
    }
    
    # Criar o plot principal (SEM informações dentro das barras)
    p = (ggplot(plot_df, aes(x='Estratégia', y='Tempo (s)', fill='Fase')) +
         geom_bar(stat='identity', position='stack', width=0.5) +
         scale_fill_manual(values=colors, name='Fase', limits=ordem_legenda) +
         scale_x_discrete(limits=['Leitura\nOtimizada', 'SIMD']) +
         scale_y_continuous(breaks=np.arange(0, 1.8 + 0.2, 0.2)) +
         labs(title='STEEP - Tempo por Fase',
              x='Algoritmo', 
              y='Tempo (s)',
              fill='Fase') +
         coord_flip() +
         theme_bw() +
         theme(axis_text_x=element_text(size=14),
               axis_text_y=element_text(size=12),
               axis_title_x=element_text(size=16),
               axis_title_y=element_text(size=16),
               plot_title=element_text(size=18, hjust=0.5),
               legend_text=element_text(size=12),
               legend_title=element_text(size=14),
               legend_position='right',
               figure_size=(16, 3)))
    
    # Salvar o plot principal limpo
    filename = "stacked_barplot_two.png"
    p.save(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"    Salvo: {output_dir / filename}")

def create_cpu_vs_gpu_stacked_barplot_with_labels(macrogroup_data, output_dir):
    """Cria stacked barplot comparando CPU vs GPU COM informações dentro das barras"""
    df, phase_mapping, phase_order = prepare_three_phase_data(macrogroup_data)
    
    if df.empty:
        print("  Aviso: Sem dados encontrados")
        return
    
    # Preparar dados para o plot - SOMANDO E_READ + E_READ_GPU e E_WRITE + E_WRITE_GPU
    plot_data = []
    
    for (strategy, algorithm), group in df.groupby(['strategy_plot', 'algorithm']):
        # Combinar E_READ e E_READ_GPU
        read_data = group[group['PHASE'].isin(['E_READ', 'E_READ_GPU'])]
        total_read_time = read_data['TIMESTAMP_median'].sum() if not read_data.empty else 0
        
        # Combinar E_WRITE e E_WRITE_GPU
        write_data = group[group['PHASE'].isin(['E_WRITE', 'E_WRITE_GPU'])]
        total_write_time = write_data['TIMESTAMP_median'].sum() if not write_data.empty else 0
        
        # Obter dados de cálculo
        calc_data = group[group['PHASE'] == 'E_CALCULO']
        total_calc_time = calc_data['TIMESTAMP_median'].sum() if not calc_data.empty else 0
        
        # Calcular tempo total para porcentagens
        total_time = total_read_time + total_calc_time + total_write_time
        
        # Adicionar dados combinados
        if total_read_time > 0:
            percentage = (total_read_time / total_time) * 100
            plot_data.append({
                'Estratégia': strategy,
                'Algoritmo': algorithm,
                'Fase': 'Leitura dos dados\nde entrada',
                'Tempo (s)': total_read_time / 1000,
                'Tempo (ms)': total_read_time,
                'Porcentagem': percentage,
                'Label': f'{total_read_time/1000:.2f}s\n({percentage:.1f}%)'
            })
        
        if total_calc_time > 0:
            percentage = (total_calc_time / total_time) * 100
            plot_data.append({
                'Estratégia': strategy,
                'Algoritmo': algorithm,
                'Fase': 'Processamento',
                'Tempo (s)': total_calc_time / 1000,
                'Tempo (ms)': total_calc_time,
                'Porcentagem': percentage,
                'Label': f'{total_calc_time/1000:.2f}s\n({percentage:.1f}%)'
            })
        
        if total_write_time > 0:
            percentage = (total_write_time / total_time) * 100
            plot_data.append({
                'Estratégia': strategy,
                'Algoritmo': algorithm,
                'Fase': 'Escrita dos dados\nde saída',
                'Tempo (s)': total_write_time / 1000,
                'Tempo (ms)': total_write_time,
                'Porcentagem': percentage,
                'Label': f'{total_write_time/1000:.2f}s\n({percentage:.1f}%)'
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Converter para Categorical para manter a ordem
    plot_df['Fase'] = pd.Categorical(plot_df['Fase'], categories=phase_order, ordered=True)
    
    # A ordem da legenda deve ser INVERSA à ordem das barras
    # Isto garante que a primeira cor na legenda corresponde à barra no topo
    ordem_legenda = list(reversed(phase_order))
    
    # Cores padrão para gráficos combinados
    colors = {
        'Escrita dos dados\nde saída': '#bdbfbf',      # Cinza
        'Processamento': '#4ECDC4',                     # Azul médio
        'Leitura dos dados\nde entrada': '#00bdbd'     # Azul forte
    }
    
    # Criar o plot principal COM informações dentro das barras
    p = (ggplot(plot_df, aes(x='Estratégia', y='Tempo (s)', fill='Fase')) +
         geom_bar(stat='identity', position='stack', width=0.5) +
         geom_text(aes(label='Label'), 
                  position=position_stack(vjust=0.5), 
                  size=10, 
                  color='black',
                  fontweight='bold') +
         scale_fill_manual(values=colors, name='Fase', limits=ordem_legenda) +
         scale_x_discrete(limits=['Leitura\nOtimizada','SIMD']) +
         scale_y_continuous(breaks=np.arange(0, 1.8 + 0.2, 0.2)) +
         labs(title='STEEP - Tempo por Fase',
              x='Algoritmo', 
              y='Tempo (s)',
              fill='Fase') +
         coord_flip() +
         theme_bw() +
         theme(axis_text_x=element_text(size=14),
               axis_text_y=element_text(size=12),
               axis_title_x=element_text(size=16),
               axis_title_y=element_text(size=16),
               plot_title=element_text(size=18, hjust=0.5),
               legend_text=element_text(size=12),
               legend_title=element_text(size=14),
               legend_position='right',
               figure_size=(16, 3)))
    
    # Salvar o plot com labels
    filename = "stacked_barplot_two_with_labels.png"
    p.save(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"    Salvo: {output_dir / filename}")

def main():
    """Função principal"""
    input_dir = Path('summarized_results_grouped')
    output_dir = Path('stacked-barplot-two')
    
    # Criar diretório de saída se não existir
    output_dir.mkdir(exist_ok=True)
    
    print("=== GERAÇÃO DE STACKED BARPLOT CPU vs GPU (MODIFICADO) ===")
    print(f"Diretório de entrada: {input_dir}")
    print(f"Diretório de saída: {output_dir}")
    
    # Carregar dados das etapas
    try:
        macrogroup_data = load_macrogroup_data(input_dir)
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return
    
    print("\n=== GERANDO STACKED BARPLOT CPU vs GPU (VERSÃO LIMPA) ===")
    create_cpu_vs_gpu_stacked_barplot_clean(macrogroup_data, output_dir)
    
    print("\n=== GERANDO STACKED BARPLOT CPU vs GPU (COM LABELS) ===")
    create_cpu_vs_gpu_stacked_barplot_with_labels(macrogroup_data, output_dir)
    
    print("\n=== GERAÇÃO CONCLUÍDA ===")
    print(f"Plots salvos em: {output_dir}")

if __name__ == "__main__":
    main()
