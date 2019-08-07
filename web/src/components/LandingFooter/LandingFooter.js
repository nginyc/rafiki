import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';
import LayoutBody from '../LandingComponents/LayoutBody';
import Typography from '../LandingComponents/Typography';
import Public from "@material-ui/icons/Public"
import GithubIcon from "../../assets/GithubIcon"

const styles = theme => ({
  root: {
    display: 'flex',
    backgroundColor: theme.palette.common.white,
    // borderTop: `1px solid ${theme.palette.border}`,
  },
  layoutBody: {
    marginTop: theme.spacing(1) * 8,
    marginBottom: theme.spacing(1) * 8,
    display: 'flex',
  },
  iconsWrapper: {
    height: 120,
  },
  icons: {
    display: 'flex',
  },
  icon: {
    width: 48,
    height: 48,
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: theme.palette.primary.light,
    marginRight: theme.spacing(1),
    '&:hover': {
      backgroundColor: theme.palette.warning.dark,
    },
  },
});


function AppFooter(props) {
  const { classes } = props;

  return (
    <Typography component="footer" className={classes.root}>
      <LayoutBody className={classes.layoutBody} width="large">
        <Grid container spacing={40}>
          <Grid item xs={12}>
            <Grid
              container
              direction="column"
              justify="flex-end"
              className={classes.iconsWrapper}
              spacing={10}
            >
              <Grid item className={classes.icons}>
                <a href="https://www.comp.nus.edu.sg/~dbsystem/" target="_blank" rel="noopener noreferrer" className={classes.icon}>
                  <Public />
                </a>
                <a href="https://github.com/nginyc/rafiki" target="_blank" rel="noopener noreferrer" className={classes.icon}>
                  <GithubIcon />
                </a>
              </Grid>
              <Grid item>© 2019 Rafiki</Grid>
            </Grid>
          </Grid>
          <Grid item>
            <Typography variant="caption">
              {'Rafiki is brought to you by the team from DBsystem NUS School of Computing'}
            </Typography>
          </Grid>
        </Grid>
      </LayoutBody>
    </Typography>
  );
}

AppFooter.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default withStyles(styles)(AppFooter);
