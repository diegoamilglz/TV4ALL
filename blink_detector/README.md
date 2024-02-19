##Como usar en Windows

si no tienes el Visual studio de C++, tienes que instalarlo para que funcione dlib
##Con entorno virtual
Si quieres hacerlo en Windows, dentro de la carpeta blink_detector:
```sh
python3 -m venv .venv
.\blink_detector\.venv\Scripts\activate
```
ahora, dentro del entorno virtual podremos instalar todas las dependencias: 
```sh
pip install -r requirements.txt
```
si te da problemas, se puede instalar uno por uno. Probablemente, los que necesites van a ser:
```sh
pip install time
```

```sh
pip install numpy
```

```sh
pip install dlib
```
No os preocupeis porque de algún error con requirements.txt, hay ciertas librerias que no son esenciales en esta versión del código. 


Una vez instalado, ejecutar el programa:

```sh
python3 main.py
```
