import React from "react";
import { Link } from "react-router-dom";
// @material-ui/core components
import withStyles from "@material-ui/core/styles/withStyles";
import Icon from "@material-ui/core/Icon";
import Fab from '@material-ui/core/Fab';
import Popover from '@material-ui/core/Popover';

// core components
import GridItem from "components/Grid/GridItem.jsx";
import GridContainer from "components/Grid/GridContainer.jsx";
import Card from "components/Card/Card.jsx";
import CardHeader from "components/Card/CardHeader.jsx";
import CardBody from "components/Card/CardBody.jsx";
import TableDatasets from "../../components/Table/TableDatasets";
// routes

const styles = {
  cardCategoryWhite: {
    "&,& a,& a:hover,& a:focus": {
      color: "rgba(255,255,255,.62)",
      margin: "0",
      fontSize: "14px",
      marginTop: "0",
      marginBottom: "0"
    },
    "& a,& a:hover,& a:focus": {
      color: "#FFFFFF"
    }
  },
  cardTitleWhite: {
    color: "#FFFFFF",
    marginTop: "0px",
    minHeight: "auto",
    fontWeight: "300",
    fontFamily: "'Roboto', 'Helvetica', 'Arial', sans-serif",
    marginBottom: "3px",
    textDecoration: "none",
    "& small": {
      color: "#777",
      fontSize: "65%",
      fontWeight: "400",
      lineHeight: "1"
    }
  },
   popOver: {
     // margin: "10px",
     // backgroundColor: "rgba(255, 255, 255, 0)", 
   },

  buttonAdd: {
    //float: "right",
    //fontSize: "15px",
    //lineHeight: "15px",
    // padding: "10px",
    backgroundColor: "rgba(255, 255, 255, 0.2)",
    border: "0px",
    //borderRadius: "5px",
    "&:hover": {
        backgroundColor: "rgba(255, 255, 255, 0.5)", 
    }
  },
  buttonDetails: {
    //fontSize: "15px",
    //lineHeight: "15px",
    padding: "10px",
    borderRadius: "5px",
    border: "0px",
    backgroundColor: "rgba(255, 255, 255, 0.2)",
    "&:hover": {
        backgroundColor: "rgba(255, 255, 255, 0.5)", 
    },
  }
};

class Dataset extends React.Component {

  state = {
    anchorEl: null,
    datasets: []
  };

  handleClick = event => {
    this.setState({
      anchorEl: event.currentTarget,
    });
  };

  handleClose = () => {
    this.setState({
      anchorEl: null,
    });
  };

  async componentDidMount() {
    const { appUtils: { rafikiClient, showError } } = this.props;
    try {
      const datasets = await rafikiClient.getDatasets()
      this.setState({ datasets });
    } catch (error) {
      showError(error, 'Failed to retrieve datasets');
    }
  }

  render() {
    const classes = this.props.classes
    const { appUtils } = this.props
    const { anchorEl } = this.state;
    const open = Boolean(anchorEl);

    return (
            <GridContainer justify="center" alignContent="center">
              <GridItem xs={12} sm={12} md={12}>
                <Card>
                  <CardHeader color="info">
                    <Fab variant="extended" className={classes.buttonAdd} onClick={this.handleClick}>
                        <Icon>add_circle</Icon>New Dataset
                    </Fab>
                    <Popover className={classes.popOver}
                        id="simple-popper"
                        open={open}
                        anchorEl={anchorEl}
                        onClose={this.handleClose}
                        anchorOrigin={{ vertical: 'bottom', horizontal: 'center', }}
                        transformOrigin={{ vertical: 'top', horizontal: 'left', }}
                    >
                     <Fab variant="extended" className={classes.buttonDetails} onClick={()=>{
                         this.props.history.push("datasets/new") //route to datasets/new, there should be a better way to do this, try to rewrite the routes file.
                         this.setState({ anchorEl:null })
                     }}>
                        <Icon> add_photo_alternate</Icon>
                        Image Classification
                     </Fab>
                    </Popover>
                  </CardHeader>
                  <CardBody>
                    <Link to="/admin/datasets/details">
                    <TableDatasets
                      tableHeaderColor="primary"
                      tableHead={["ID","Name","Task", "Size", "Uploaded At"]}
                      tableData={this.state.datasets}
                      appUtils={appUtils}
                    />
                    </Link>
                  </CardBody>
                </Card>
              </GridItem>
            </GridContainer>
        );
    }
}


export default withStyles(styles)(Dataset);
