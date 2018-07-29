# Auto Tune Models Example

## Installation

1. Install Docker

## Running the Stack

Create a .env at root of project:
```
MYSQL_HOST=<docker_host>
MYSQL_PORT=3306
MYSQL_USER=atm
MYSQL_DATABASE=atm
MYSQL_PASSWORD=atm
```

Run in terminal 1:

```shell
bash scripts/start_db.sh
```

Run in terminal 2:

```shell
bash scripts/start_admin.sh
```

Run in terminal 3:

```shell
bash scripts/start_worker.sh
```

## TODO

- Add custom algorithms to ATM

## Resources
