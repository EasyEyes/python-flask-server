# python-flask-server

## Heroku deployment

This app targets the `heroku-26` stack and Python 3.12. Heroku uses the
`app.json` stack for Review Apps and Heroku CI, and `.python-version` keeps the
Python major version consistent while allowing automatic patch updates.

To upgrade the existing Cedar app, change its stack before the next deployment:

```sh
heroku stack:set heroku-26 --app easyeyes-python-flask-server
```

The next build uses Heroku-26. Confirm the deployed stack and Python version
afterward:

```sh
heroku stack --app easyeyes-python-flask-server
heroku run python --version --app easyeyes-python-flask-server
```
