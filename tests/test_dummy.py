def test_imports():
    try:
        import flask
        import numpy
        import PIL
        import tflite_runtime
    except ImportError as e:
        # Если импорт не удался, мы принудительно вызываем провал теста
        # и ищем свою причину
        assert False, f"Ошибка: библиотека не найдена ({e}). Проверь requirements.txt!"