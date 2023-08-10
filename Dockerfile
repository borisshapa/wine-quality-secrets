FROM python:3.10

SHELL ["/bin/bash", "-c"]

ARG ANSIBLE_PWD="Ansible208504AU"
ARG MSSQL_SERVER="mssql,1433"
ARG MSSQL_UID="sa"
ARG MSSQL_PWD="Droider0ru208504AU"
ARG MSSQL_DATABASE="WineQuality"

RUN pip install --upgrade pip

WORKDIR /app
COPY requirements.txt /app
COPY sh_scripts /app/sh_scripts

RUN chmod +x sh_scripts/wait_for_it.sh sh_scripts/install_odbc_debian.sh
RUN ./sh_scripts/install_odbc_debian.sh

RUN printf "server: \"$MSSQL_SERVER\"\nuid: \"$MSSQL_UID\"\npwd: \"$MSSQL_PWD\" \
    \ndatabase: \"$MSSQL_DATABASE\"" >> mssql-creds.yml

RUN printf $ANSIBLE_PWD >> ansible-pwd.txt

RUN python3 -m pip install --user ansible
RUN printf "export PATH=$HOME/.local/bin:$PATH\n" >> ~/.bashrc
RUN source ~/.bashrc && ansible-vault encrypt mssql-creds.yml --vault-password-file ansible-pwd.txt

RUN pip install -r requirements.txt

ENTRYPOINT ["bash"]
