/*
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at
 
    http://www.apache.org/licenses/LICENSE-2.0
 
  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
 */

import * as React from 'react';
import { withStyles } from '@material-ui/core/styles';
import { Typography, Paper, FormControl, 
  Input, InputLabel, Button } from '@material-ui/core';

import { AppRoute } from '../app/AppNavigator';

/*interface Props {
  classes: { [s: string]: any };
  appUtils: AppUtils;
}*/

class LoginPage extends React.Component {
  state = {
    email: '',
    password: ''
  }

  async onLoginSubmit(e) {
    const { appUtils } = this.props; /* appUtils: { rafikiClient, appNavigator, showError } */
    const { email, password } = this.state;

    e.preventDefault();
    
    try {
      await appUtils.rafikiClient.login(email, password);
      appUtils.appNavigator.goTo(AppRoute.DATASETS);
    } catch (error) {
      appUtils.showError(error, 'Failed to Login');
    }
  }

  async onChange(name, e) {
    const value = e.target.value;
    this.setState({ [name]: value });
  }

  render() {
    const { classes } = this.props;
    const { email, password } = this.state;

    return (
      <main className={classes.main}>
          <Paper className={classes.loginPaper}>
            <Typography variant="h3">Login</Typography>
            <form className={classes.loginForm} onSubmit={e => this.onLoginSubmit(e)}>
              <FormControl margin="normal" required fullWidth>
                <InputLabel htmlFor="email">Email Address</InputLabel>
                <Input name="email" autoComplete="email" autoFocus value={email} 
                  onChange={e => this.onChange('email', e)} />
              </FormControl>
              <FormControl margin="normal" required fullWidth>
                <InputLabel htmlFor="password">Password</InputLabel>
                <Input value={password} name="password" type="password" autoComplete="current-password"
                  onChange={e => this.onChange('password', e)} />
              </FormControl>
              <Button className={classes.loginBtn} type="submit" fullWidth color="primary">
                Login
              </Button>
            </form>
          </Paper>
      </main>
    );
  }
}

const styles = (theme) => ({
  main: {
    margin: theme.spacing.unit * 8,
    [theme.breakpoints.up('md')]: {
      width: 400,
      marginLeft: 'auto',
      marginRight: 'auto',
    },
  },
  loginPaper: {
    padding: theme.spacing.unit * 4
  },
  loginBtn: {
    marginTop: theme.spacing.unit * 2
  }
});

export default withStyles(styles)(LoginPage);