from app import create_app
from app.extensions import sock

app = create_app()

if __name__ == '__main__':
    # Print all the available routes
    print(app.url_map)

    app.run(host='0.0.0.0')
    # sock.run(app)

