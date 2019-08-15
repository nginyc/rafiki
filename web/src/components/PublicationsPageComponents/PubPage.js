import React from 'react';
import { withStyles } from '@material-ui/core/styles';

import LayoutBody from '../LandingComponents/LayoutBody';
import Typography from '../LandingComponents/Typography';

import '../LandingMainPage/Overview.css'


const styles = theme => ({
  root: {
    marginTop: theme.spacing(1) * 8,
  },
})


function PubPage(props) {
  const { classes } = props;
  return (
    <React.Fragment>
      <LayoutBody className={classes.root} component="section" width="large">
        <Typography variant="h3" gutterBottom marked="center" align="center">
          Publications
        </Typography>
        <div className="section_center">
          <div className="center__description">
            <p className="description__text">
              <b>Rafiki: Machine Learning as an Analytics Service System. </b>
Wei Wang, Sheng Wang, Jinyang Gao, Meihui Zhang, Gang Chen, Teck Khim Ng, Beng Chin Ooi, and Jie Shao. Accepted in VLDB 2019. [<a href="https://arxiv.org/abs/1804.06087f" target="_blank" rel="noopener noreferrer"> paper </a>] 
            </p>
          </div>
          </div>
      </LayoutBody>
    </React.Fragment>
      );
    }
    
    export default withStyles(styles)(PubPage);
