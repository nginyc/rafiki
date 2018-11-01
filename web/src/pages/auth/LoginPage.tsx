import * as React from 'react';
import { withStyles, StyleRulesCallback } from '@material-ui/core/styles';
import { Typography, Paper, FormControl, 
  Input, InputLabel, Button } from '@material-ui/core';

import { AppUtils } from '../../App';
import { AppRoute } from '../../app/AppNavigator';

interface Props {
  classes: { [s: string]: any };
  appUtils: AppUtils;
}

class LoginPage extends React.Component<Props> {
  state = {
    email: '',
    password: ''
  }

  async onLoginSubmit(e: React.FormEvent) {
    const { appUtils: { rafikiClient, appNavigator, showError } } = this.props;
    const { email, password } = this.state;

    e.preventDefault();
    
    try {
      await rafikiClient.login(email, password);
      appNavigator.goTo(AppRoute.DASHBOARD);
    } catch (error) {
      showError(error, 'Failed to Login');
    }
  }

  async onChange(name: string, e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) {
    const value = e.target.value;
    this.setState({ [name]: value });
  }

  render() {
    const { classes, appUtils } = this.props;
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

const styles: StyleRulesCallback = (theme) => ({
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