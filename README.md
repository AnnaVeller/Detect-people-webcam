# Detect-people-webcam

### Программа для распознавания лиц

*Программа используется для нахождения и распознавания людей.*

------------


#### Для запуска программы неоходимо выполнить:

```
pip install opencv-python

pip install numpy

pip install argparse

pip install face_recognition
```
Последнее может не выполниться 

-> решение для Windows: `pip install cmake `


#### Функционал

Программа распознает лица в режиме реального времени по вебкамере. Обводит в рамку и выделяет губы синим.

Можно создать папку на компьютере "people". В ней создать папки с именами людей, чьи фото поместить внутрь. Тогда при распознавании лиц, программа также будет искать этих людей и при нахождении подписывать имя.

При запуске программы можно указать параметр --debug и указать уровень выведения логов. Для более подробного описания при вызове программы укажите параметр -h.

***Основной функционал программы был написан благодаря <https://github.com/ageitgey/face_recognition>***
