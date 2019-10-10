import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import { Link } from "react-router-dom";
import Button from '../LandingComponents/Button';
import Typography from '../LandingComponents/Typography';
import ProductHeroLayout from './LandingProductHeroLayout';
import heroImage from "../../assets/electrical-2476782_960_720.jpg"

const backgroundImage = heroImage
  //'https://www.ebetfinder.com/wp-content/uploads/2016/11/dota-2-gameplay-ebetfinder-resized.jpg';
  //'https://www.comp.nus.edu.sg/~dbsystem/rafiki/pic/stack.jpg'
  //'https://www.geek.com/wp-content/uploads/2010/12/asfMod_02.jpg'
  //'https://cdn.pixabay.com/photo/2017/07/06/03/00/electrical-2476782_960_720.jpg'

const styles = theme => ({
  background: {
    backgroundImage: `url(${heroImage})`,
    backgroundColor: '#333333', // Average color of the background image.
    backgroundPosition: 'center',
  },
  button: {
    //minWidth: 200,
  },
  h5: {
    marginBottom: theme.spacing(1) * 4,
    marginTop: theme.spacing(1) * 4,
    [theme.breakpoints.up('sm')]: {
      marginTop: theme.spacing(1) * 8,
    },
  },
  more: {
    marginTop: theme.spacing(1) * 2,
  },
});

function ProductHero(props) {
  const { classes } = props;

  return (
    <ProductHeroLayout backgroundClassName={classes.background}>
      {/* Increase the network loading priority of the background image. */}
      <img style={{ display: 'none' }} src={backgroundImage} alt="" />
      <Typography color="inherit" align="center" variant="h2" marked="center">
        Rafiki
      </Typography>
      <Typography color="inherit" align="center" variant="h5" className={classes.h5}>
        Rafiki is a distributed system that trains machine learning (ML) models and deploys trained models, built with ease-of-use in mind. 
      </Typography>
      <Button
        color="secondary"
        variant="contained"
        size="large"
        className={classes.button}
        component={Link}
        to={`/console`}
      >
        Try Rafiki
      </Button>
      <Typography variant="body2" color="inherit" className={classes.more}>
        Discover the experience
      </Typography>
    </ProductHeroLayout>
  );
}

ProductHero.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default withStyles(styles)(ProductHero);