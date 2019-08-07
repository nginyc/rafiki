import React from 'react';
import { Link } from "react-router-dom";

import { withStyles } from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';
import Hidden from '@material-ui/core/Hidden';
import Typography from '../LandingComponents/Typography';
import LayoutBody from '../LandingComponents/LayoutBody';
import Button from '../LandingComponents/Button';

import tryRafiki from "../../assets/tryRafiki.png"

const styles = theme => ({
  root: {
    marginTop: theme.spacing(1) * 10,
    //marginBottom: 0,
    marginBottom: theme.spacing(1) * 9,
    display: 'flex',
  },
  cardWrapper: {
    zIndex: 1,
  },
  card: {
    display: 'flex',
    justifyContent: 'center',
    padding: `${theme.spacing(1) * 8}px ${theme.spacing(1) * 3}px`,
  },
  cardContent: {
    maxWidth: 400,
  },
  button: {
    width: '100%',
  },
  imagesWrapper: {
    position: 'relative',
  },
  image: {
    position: 'absolute',
    top: -28,
    left: -28,
    right: 0,
    bottom: 0,
    width: '100%',
    maxWidth: 600,
  },
});

class LandingTryRafiki extends React.Component {
  render() {
    const { classes } = this.props;

    return (
      <LayoutBody className={classes.root} component="section" width="large">
        <Grid container spacing={0}>
          <Grid item xs={12} md={6} className={classes.cardWrapper}>
            <div className={classes.card}>
              <div className={classes.cardContent}>
                <Typography variant="h2" component="h2" gutterBottom>
                  Try rafiki
                </Typography>
                <Typography variant="h5">
                a distributed system that trains machine learning (ML) models and deploys trained models
                </Typography>
                <br />
                <Button
                  type="submit"
                  color="primary"
                  variant="contained"
                  className={classes.button}
                  component={Link}
                  to={`/contact`}
                >
                  Find and Contact Us
                </Button>
              </div>
            </div>
          </Grid>
          <Grid item xs={12} md={6} className={classes.imagesWrapper}>
            <Hidden smDown>
              <img
                src={tryRafiki}
                alt="tryRafiki"
                className={classes.image}
              />
            </Hidden>
          </Grid>
        </Grid>
      </LayoutBody>
    );
  }
}

export default withStyles(styles)(LandingTryRafiki);
