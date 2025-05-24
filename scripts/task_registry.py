# task_registry.py

TASK_HANDLERS = {}

def task(name):
    """
    Функция декоратор
    Атрибуты:
    name - название задачи
    """
    def wrapper(func):
        TASK_HANDLERS[name] = func
        return func
    return wrapper

def main(task_names):
    """
        Функция обработки задач для DVC из списка  TASK_HANDLERS
        Атрибуты:
        task_names - название задачи
    """
    for task_name in task_names:
        if task_name not in TASK_HANDLERS:
            raise ValueError(f"Неизвестная задача: {task_name}\nДоступные задачи: {list(TASK_HANDLERS.keys())}")
        print(f"\n=== Выполняется задача: {task_name} ===")
        TASK_HANDLERS[task_name]()