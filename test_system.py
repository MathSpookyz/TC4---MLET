"""
Script de teste para verificar o funcionamento do sistema atualizado
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "scrapper"))
sys.path.append(str(Path(__file__).parent / "model"))

def test_scrapper():
    """Testa o pipeline de scraping"""
    print("=" * 60)
    print("TESTE 1: Pipeline de Scraping")
    print("=" * 60)
    
    try:
        from scrapper.scrapper_pipeline import get_or_scrappe_ticker
        
        ticker = "PETR4.SA"
        print(f"\nBuscando dados para {ticker}...")
        
        df = get_or_scrappe_ticker(
            ticker=ticker,
            data_path="scrapper/data/processed/prices"
        )
        
        if df is not None and not df.empty:
            print(f"✓ Dados obtidos com sucesso!")
            print(f"  - Registros: {len(df)}")
            print(f"  - Colunas: {list(df.columns)}")
            print(f"  - Período: {df['date'].min()} até {df['date'].max()}")
            print(f"\nPrimeiras linhas:")
            print(df.head())
            return True
        else:
            print("✗ Falha ao obter dados")
            return False
            
    except Exception as e:
        print(f"✗ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training():
    """Testa o treinamento de modelo"""
    print("\n" + "=" * 60)
    print("TESTE 2: Treinamento de Modelo")
    print("=" * 60)
    
    try:
        from model.model_training import train_model
        
        ticker = "PETR4.SA"
        print(f"\nTreinando modelo para {ticker}...")
        print("(Este processo pode levar alguns minutos)\n")
        
        result = train_model(
            ticker=ticker,
            start_date="2023-01-01"
        )
        
        if result:
            print(f"\n✓ Modelo treinado com sucesso!")
            print(f"  - RMSE: R$ {result['rmse']:.2f}")
            print(f"  - Última previsão: R$ {result['next_prediction']:.2f}")
            print(f"  - Modelo salvo em: {result['model_path']}")
            return True
        else:
            print("✗ Falha no treinamento")
            return False
            
    except Exception as e:
        print(f"✗ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prediction():
    """Testa a previsão com modelo treinado"""
    print("\n" + "=" * 60)
    print("TESTE 3: Previsão de Preços")
    print("=" * 60)
    
    try:
        from scrapper.scrapper_pipeline import get_or_scrappe_ticker
        from model_executor import predict_price
        
        ticker = "PETR4.SA"
        print(f"\nFazendo previsão para {ticker}...")
        
        # Obter dados
        df = get_or_scrappe_ticker(
            ticker=ticker,
            data_path="scrapper/data/processed/prices"
        )
        
        if df is None or df.empty:
            print("✗ Dados não disponíveis")
            return False
        
        # Fazer previsão
        result = predict_price(df, ticker=ticker, days=5)
        
        print(f"\n✓ Previsão realizada com sucesso!")
        print(f"  - Último preço conhecido: R$ {result['last_known_price']:.2f}")
        print(f"  - Previsões para os próximos {result['days']} dias:")
        for i, pred in enumerate(result['predictions'], 1):
            print(f"    Dia {i}: R$ {pred:.2f}")
        
        return True
            
    except FileNotFoundError as e:
        print(f"✗ Modelo não encontrado: {e}")
        print("  Execute primeiro o TESTE 2 (treinamento)")
        return False
    except Exception as e:
        print(f"✗ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Executa todos os testes"""
    print("\n" + "=" * 60)
    print("TESTE DO SISTEMA DE PREVISÃO DE AÇÕES")
    print("=" * 60)
    print("\nEste script irá testar:")
    print("1. Pipeline de scraping de dados")
    print("2. Treinamento de modelo")
    print("3. Previsão de preços")
    print("\n" + "=" * 60)
    
    input("\nPressione ENTER para iniciar os testes...")
    
    results = []
    
    results.append(("Scraping", test_scrapper()))
    
    print("\n")
    resposta = input("Deseja executar o treinamento? (s/n): ").strip().lower()
    if resposta == 's':
        results.append(("Treinamento", test_training()))
    else:
        print("Teste de treinamento pulado.")
        results.append(("Treinamento", None))
    
    results.append(("Previsão", test_prediction()))
    
    print("\n" + "=" * 60)
    print("RESUMO DOS TESTES")
    print("=" * 60)
    
    for name, result in results:
        if result is True:
            status = "✓ PASSOU"
        elif result is False:
            status = "✗ FALHOU"
        else:
            status = "- PULADO"
        print(f"{name:20} {status}")
    
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r is True)
    total = sum(1 for _, r in results if r is not None)
    
    if passed == total and total > 0:
        print("\n🎉 Todos os testes passaram!")
    else:
        print(f"\n⚠️  {passed}/{total} testes passaram")


if __name__ == "__main__":
    main()
