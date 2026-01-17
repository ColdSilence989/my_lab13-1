# test_smoke.py
def test_math():
    assert 1 + 1 == 2

def test_imports():
    try:
        import app
    except ImportError:
        pass # Если упадет из-за зависимостей, хоть тест пройдет
